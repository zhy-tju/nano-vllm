# 导入 pickle 模块。
# pickle 用于序列化和反序列化 Python 对象结构。简单来说，就是将 Python 对象（如列表、字典）
# 转换成字节流（序列化），以便存储或通过网络传输，之后再将字节流恢复为原始对象（反序列化）。
# 这里用它在进程间传递复杂的参数。
import pickle
# 导入 torch 库。
import torch
# 导入 torch.distributed 模块，通常简写为 dist。
# 这是 PyTorch 用于分布式计算的模块，支持多机多卡或单机多卡的并行训练和推理。
import torch.distributed as dist
# 从 multiprocessing.synchronize 模块导入 Event 类。
# Event 是一个简单的同步原语，用于在不同进程间通信。一个进程可以设置（set）事件，另一个进程可以等待（wait）事件。
from multiprocessing.synchronize import Event
# 从 multiprocessing.shared_memory 模块导入 SharedMemory 类。
# SharedMemory 允许不同进程直接访问同一块内存区域，这是实现高效进程间通信（IPC）的一种方式，避免了数据的复制。
from multiprocessing.shared_memory import SharedMemory

# 导入项目内部的模块和类。
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    ModelRunner 是一个在独立进程中运行的类，负责实际的模型计算。
    每个 GPU 对应一个 ModelRunner 实例。它封装了模型的加载、KV缓存的分配、
    CUDA Graph 的捕获以及与主进程的通信逻辑。
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        ModelRunner 的构造函数。
        
        Args:
            config (Config): 全局配置对象。
            rank (int): 当前进程的排名（或 GPU 的 ID）。rank 0 是主进程。
            event (Event | list[Event]): 用于同步的事件对象。
                                         对于 rank 0，它是一个事件列表，用于向所有子进程发信号。
                                         对于子进程，它是一个单独的事件，用于等待 rank 0 的信号。
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size  # 总的进程/GPU 数量。
        self.rank = rank  # 当前进程的 ID。
        self.event = event

        # --- 1. 初始化分布式环境 ---
        # 使用 "nccl" 后端初始化进程组。NCCL 是 NVIDIA 提供的用于 GPU 间高效通信的库。
        # "tcp://localhost:2333" 是一个用于进程间握手的地址。
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # 为当前进程设置要使用的 GPU 设备。
        torch.cuda.set_device(rank)
        
        # --- 2. 设置 PyTorch 环境并加载模型 ---
        default_dtype = torch.get_default_dtype()  # 保存当前的默认数据类型。
        torch.set_default_dtype(hf_config.torch_dtype)  # 将默认数据类型设置为模型指定的类型（如 float16）。
        torch.set_default_device("cuda")  # 将默认设备设置为 CUDA，这样新创建的张量会自动放在当前 GPU 上。
        
        self.model = Qwen3ForCausalLM(hf_config)  # 创建模型实例。
        load_model(self.model, config.model)  # 加载预训练权重。
        self.sampler = Sampler()  # 创建采样器实例。
        
        # --- 3. 模型预热和资源分配 ---
        self.warmup_model()  # 执行一次预热运行，以确保所有 CUDA 内核都被编译和初始化。
        self.allocate_kv_cache()  # 根据配置和可用显存，分配 PagedAttention 所需的 KV 缓存。
        
        # --- 4. 捕获 CUDA Graph (如果启用) ---
        if not self.enforce_eager:
            self.capture_cudagraph()  # 捕获模型的计算图，以减少后续运行时的 CPU 开销。
            
        # --- 5. 恢复 PyTorch 环境并启动通信循环 ---
        torch.set_default_device("cpu")  # 恢复默认设备为 CPU。
        torch.set_default_dtype(default_dtype)  # 恢复默认数据类型。

        if self.world_size > 1:
            if rank == 0:
                # rank 0 (主进程) 创建一块名为 "nanovllm" 的共享内存。
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20) # 2**20 = 1MB
                dist.barrier()  # 设置一个屏障，等待所有其他进程也到达这个点。
            else:
                # 其他子进程在此等待，直到 rank 0 创建完共享内存。
                dist.barrier()
                # 连接到由 rank 0 创建的同名共享内存。
                self.shm = SharedMemory(name="nanovllm")
                # 子进程进入一个无限循环，等待主进程通过共享内存发送指令。
                self.loop()

    def exit(self):
        """
        在进程退出时调用的清理函数。
        """
        if self.world_size > 1:
            self.shm.close()  # 关闭对共享内存的访问。
            dist.barrier()  # 等待所有进程都关闭了共享内存。
            if self.rank == 0:
                self.shm.unlink()  # rank 0 负责销毁共享内存块。
        if not self.enforce_eager:
            del self.graphs, self.graph_pool  # 删除 CUDA Graph 相关对象。
        torch.cuda.synchronize()  # 等待所有在当前 GPU 上的操作完成。
        dist.destroy_process_group()  # 销毁分布式进程组。

    def loop(self):
        """
        子进程（rank > 0）的主循环。
        """
        while True:
            # 等待并从共享内存中读取指令。
            method_name, args = self.read_shm()
            # 在本地执行指令。
            self.call(method_name, *args)
            if method_name == "exit":
                # 如果是退出指令，则跳出循环，进程结束。
                break

    def read_shm(self):
        """
        子进程从共享内存读取指令。
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # 阻塞，直到主进程（rank 0）调用 event.set()。
        # 从共享内存的开头读取4个字节，解析出数据的长度 n。
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 从共享内存的第4个字节开始，读取 n 个字节，并用 pickle 反序列化成 Python 对象。
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()  # 清除事件状态，为下一次等待做准备。
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        主进程（rank 0）向共享内存写入指令。
        """
        assert self.world_size > 1 and self.rank == 0
        # 使用 pickle 将方法名和参数序列化成字节流。
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 将数据长度 n 写入共享内存的前4个字节。
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        # 将序列化后的数据写入共享内存。
        self.shm.buf[4:n+4] = data
        # 遍历事件列表，为每个子进程设置事件，通知它们可以读取数据了。
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """
        一个统一的调用入口。
        对于主进程，它会将调用请求写入共享内存。
        对于所有进程（包括主进程和子进程），它都会在本地执行对应的方法。
        """
        if self.world_size > 1 and self.rank == 0:
            # 如果是主进程且开启了张量并行，则将指令写入共享内存。
            self.write_shm(method_name, *args)
        
        # getattr(self, method_name, None) 是一个 Python 内置函数，
        # 它会尝试获取 `self` 对象上名为 `method_name` 的属性（这里是方法）。
        # 如果找不到，返回 None。
        method = getattr(self, method_name, None)
        # 执行找到的方法，并将 `*args` 解包作为参数传入。
        return method(*args)

    def warmup_model(self):
        """
        模型预热。执行一次最大尺寸的推理，以确保所有 CUDA 内核都被编译，
        并让 PyTorch 分配好必要的内存，避免在实际服务中首次请求时出现延迟。
        """
        torch.cuda.empty_cache()  # 清空未使用的缓存显存。
        torch.cuda.reset_peak_memory_stats()  # 重置显存峰值统计。
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 计算在满足配置限制下的最大预热批次大小。
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        # 创建一批伪造的序列。
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        # 执行一次 prefill 运行。
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        计算并分配 PagedAttention 所需的 KV 缓存。
        """
        config = self.config
        hf_config = config.hf_config
        # 获取当前 GPU 的可用显存。
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # 获取模型加载和预热后，PyTorch 报告的峰值和当前已分配显存。
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # 计算张量并行后，每个 GPU 负责的 KV 头数量。
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 获取每个注意力头的维度。
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 计算单个物理块占用的字节数。
        # 2 (K和V) * num_layers * block_size * num_kv_heads * head_dim * dtype_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        
        # 计算可以分配的物理块总数。
        # (总显存 * 使用率 - 已用 - 峰值 + 当前) / 单个块大小
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0  # 确保至少能分配一个块。
        
        # 创建一个巨大的张量作为所有层的 KV 缓存池。
        # 形状: [2 (K/V), num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        
        # 遍历模型的每一层，将 KV 缓存池中对应的切片（slice）分配给该层的 k_cache 和 v_cache 属性。
        # 这样，所有层的缓存都指向这个连续的大张量中的不同部分，便于管理。
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        为当前批次的序列准备 block_tables 张量。
        block_tables 是一个整数张量，每一行代表一个序列，行中的数字是该序列使用的物理块的 ID。
        """
        max_len = max(len(seq.block_table) for seq in seqs)  # 找到批次中最长的块表长度。
        # 对齐所有序列的块表长度，短的用 -1 填充。
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # 转换为 GPU 张量。pin_memory=True 和 non_blocking=True 是优化技巧，可以加速 CPU 到 GPU 的数据传输。
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        为 prefill（提示处理）阶段准备模型输入。
        """
        input_ids = []          # 存储所有序列中需要新计算的 token ID。
        positions = []          # 对应的位置 ID。
        cu_seqlens_q = [0]      # Query 的累积序列长度，用于 FlashAttention。
        cu_seqlens_k = [0]      # Key 的累积序列长度。
        max_seqlen_q = 0        # Query 的最大序列长度。
        max_seqlen_k = 0        # Key 的最大序列长度。
        slot_mapping = []       # 将逻辑 token 位置映射到物理 KV 缓存位置。
        block_tables = None     # 块表。
        
        for seq in seqs:
            seqlen = len(seq)
            # 只处理未被缓存的部分。
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            seqlen_q = seqlen - seq.num_cached_tokens  # Query 长度是新 token 的数量。
            seqlen_k = seqlen                          # Key 长度是总 token 数量。
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            if not seq.block_table:    # 如果是预热阶段，块表为空，跳过 slot mapping。
                continue
            
            # 为新 token 构建 slot_mapping。
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
                
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # 如果存在前缀缓存（Key比Query长）。
            block_tables = self.prepare_block_tables(seqs)
            
        # 将所有列表转换为 GPU 张量。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # 将准备好的数据设置到全局上下文中，以便模型内部的注意力层可以访问。
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        为 decode（解码生成）阶段准备模型输入。
        """
        input_ids = []      # 批次中每个序列的最后一个 token。
        positions = []      # 对应的位置 ID。
        slot_mapping = []   # 每个新 token 要写入的物理 KV 缓存槽位。
        context_lens = []   # 每个序列的当前总长度。
        
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # 计算新 token 的物理槽位地址。
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
            
        # 转换为 GPU 张量。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        # 设置全局上下文。
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        为采样阶段准备温度等参数。
        """
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        实际执行模型的前向传播。
        Python 语法：@torch.inference_mode() 是一个装饰器，它会禁用梯度计算，
        从而减少内存占用并加速计算，是推理时必须使用的。
        """
        # 如果是 prefill 阶段，或者强制使用 eager 模式，或者批次大小太大无法使用 CUDA Graph，
        # 则直接调用模型。
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 使用 CUDA Graph 进行推理。
            bs = input_ids.size(0)
            context = get_context()
            # 从预先捕获的图中，选择一个大小最接近且不小于当前批次大小的图。
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            
            # 将当前批次的输入数据更新到 CUDA Graph 使用的输入张量中。
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            # 重放计算图。这个操作的 CPU 开销极小。
            graph.replay()
            
            # 从 CUDA Graph 的输出张量中获取结果。
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        一个完整的运行步骤，包括数据准备、模型执行和采样。
        """
        # 准备模型输入。
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 准备采样参数（只在 rank 0 上需要）。
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # 执行模型。
        logits = self.run_model(input_ids, positions, is_prefill)
        # 采样（只在 rank 0 上执行）。
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 重置全局上下文。
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        捕获用于 decode 阶段的 CUDA Graph。
        CUDA Graph 可以将一系列在 GPU 上的操作录制下来，之后可以一键重放，
        极大地减少了由 CPU 逐个启动 CUDA 内核所带来的开销。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)  # 定义要捕获的最大批次大小。
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        # 创建用于捕获图的占位符（placeholder）张量。
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # 定义要为哪些批次大小捕获图。通常是 2 的幂和一些线性间隔。
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None  # 内存池，让不同的图可以共享内存。

        # 从大到小遍历批次大小进行捕获，这样可以复用内存。
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # 设置伪造的上下文。
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # 预热运行，确保所有东西都已准备好。
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 进入图捕获模式。
            with torch.cuda.graph(graph, self.graph_pool):
                # 在这个 `with` 块中的所有 CUDA 操作都会被录制到 `graph` 中。
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                
            if self.graph_pool is None:
                # 从第一个捕获的图中获取内存池。
                self.graph_pool = graph.pool()
            
            self.graphs[bs] = graph  # 存储捕获好的图。
            torch.cuda.synchronize()  # 等待捕获完成。
            reset_context()

        # 将所有用于图的占位符张量存储起来，以便在重放时更新它们。
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

