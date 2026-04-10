# 导入 torch 库
import torch
# 从 torch 模块导入 nn
from torch import nn
# 从 torch.nn 模块导入 functional，简写为 F
import torch.nn.functional as F
# 导入 torch.distributed 模块，简写为 dist，用于分布式计算
import torch.distributed as dist

# 导入项目工具
from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    实现了词表并行（Vocabulary Parallelism）的 Embedding 层。
    在张量并行中，当词表非常大时（例如几十万），将 Embedding 矩阵按列切分到不同的 GPU 上可以节省显存。
    每个 GPU 只存储词表的一部分。
    """

    def __init__(
        self,
        num_embeddings: int,  # 词表的总大小
        embedding_dim: int,   # embedding 向量的维度
    ):
        super().__init__()
        # 获取当前 GPU 在分布式环境中的排名（rank）和总大小（world_size）。
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # 断言：确保词表大小可以被并行规模整除。
        assert num_embeddings % self.tp_size == 0
        
        self.num_embeddings = num_embeddings
        # 计算每个 GPU 分区（partition）负责的词表大小。
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        
        # 计算当前 GPU 负责的词表索引范围。
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
        # 创建当前 GPU 分区对应的权重参数。注意，大小只是完整词表的一部分。
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        
        # Python 语法：动态地给一个对象添加属性。
        # 这里我们将一个自定义的加载函数 `self.weight_loader` 附加到 `self.weight` 参数对象上。
        # 这样做使得 `utils/loader.py` 中的加载逻辑可以发现并使用这个自定义加载器。
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        自定义的权重加载器，用于正确地从完整的权重文件中加载当前 GPU 需要的分片。
        """
        param_data = param.data
        shard_size = param_data.size(0)  # 当前分片的大小。
        # 计算当前分片在完整权重张量中的起始索引。
        start_idx = self.tp_rank * shard_size
        # 从加载的完整权重中，切出（narrow）当前 GPU 需要的部分。
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        # 将切片复制到参数中。
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        前向传播。
        
        Args:
            x (torch.Tensor): 输入的 token ID 张量。
        """
        if self.tp_size > 1:
            # --- 1. 掩码和索引转换 (Mask and Transform) ---
            # 创建一个布尔掩码，标记出哪些输入的 token ID 属于当前 GPU 的管辖范围。
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将全局的 token ID 转换成本地的 embedding 索引。
            # 不在范围内的 token ID 会被乘以 0，变成 0，但它们对应的 embedding 结果稍后也会被 mask 掉。
            x = mask * (x - self.vocab_start_idx)
            
        # --- 2. 本地 Embedding (Local Embedding) ---
        # 使用 F.embedding 在本地的权重分片上进行查找。
        y = F.embedding(x, self.weight)
        
        if self.tp_size > 1:
            # --- 3. 结果聚合 (Aggregate Results) ---
            # 将不属于当前 GPU 的 token 对应的 embedding 结果置为 0。
            # unsqueeze(1) 在维度1上增加一个维度，以匹配 y 的形状进行广播。
            y = mask.unsqueeze(1) * y
            # 使用 all_reduce 操作将所有 GPU 上的 embedding 结果相加。
            # 由于每个 token ID 只在一个 GPU 上有非零的 embedding 结果，
            # all_reduce 之后，每个 GPU 上的 y 张量都会包含最终的、完整的 embedding 结果。
            dist.all_reduce(y)
            
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    实现了词表并行的 LM Head (语言模型头)。
    LM Head 通常是模型最后一层，它将模型的输出隐状态投影回词表空间，得到每个 token 的 logits。
    它的权重矩阵实际上是与 Embedding 层的权重矩阵转置后共享的（或类似）。
    因此，它也需要进行词表并行。这个类直接继承了 VocabParallelEmbedding 以复用其并行逻辑。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias  # 这个实现不支持偏置项。
        # 调用父类的构造函数，完成权重的分区等初始化。
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        前向传播。
        
        Args:
            x (torch.Tensor): 来自模型最后一层 Transformer Block 的输出隐状态。
        """
        context = get_context()
        if context.is_prefill:
            # 在 prefill 阶段，输入 x 包含了所有 token 的隐状态。
            # 但我们只需要计算每个序列最后一个 token 的 logits 来进行采样。
            # `context.cu_seqlens_q` 存储了累积长度，所以 `cu_seqlens_q[1:] - 1` 就得到了每个序列最后一个 token 的索引。
            last_indices = context.cu_seqlens_q[1:] - 1
            # 从 x 中抽取出这些最后一个 token 的隐状态。
            # .contiguous() 确保张量在内存中是连续的，这对于后续操作的性能很重要。
            x = x[last_indices].contiguous()
            
        # --- 1. 本地 Logits 计算 (Local Logits) ---
        # F.linear(x, self.weight) 执行矩阵乘法 x @ weight.T。
        # 由于 self.weight 只是词表的一部分，这里计算出的 logits 也只是部分 logits。
        logits = F.linear(x, self.weight)
        
        if self.tp_size > 1:
            # --- 2. 结果聚合 (Aggregate Results) ---
            # 在 rank 0 上创建一个列表，用于接收来自所有 GPU 的部分 logits。
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 使用 gather 操作将所有 GPU 上的 logits 收集到 rank 0 的 `all_logits` 列表中。
            dist.gather(logits, all_logits, 0)
            
            if self.tp_rank == 0:
                # 在 rank 0 上，将收集到的所有部分 logits 沿最后一个维度拼接起来，形成完整的 logits。
                logits = torch.cat(all_logits, -1)
            else:
                # 在其他 rank 上，logits 为 None，因为采样只在 rank 0 上进行。
                logits = None
                
        return logits

