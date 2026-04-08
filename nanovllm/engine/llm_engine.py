# 导入 atexit 模块。
# atexit 允许你注册一个或多个函数，这些函数将在 Python 解释器正常终止时自动执行。
# 这对于执行清理操作（如关闭子进程、保存状态等）非常有用。
import atexit
# 从 dataclasses 模块导入 fields 函数。
# fields(dataclass) 会返回一个元组，其中包含该 dataclass 的所有字段对象，可以用来获取字段名、类型等信息。
from dataclasses import fields
# 从 time 模块导入 perf_counter 函数。
# perf_counter 提供了一个高精度的性能计数器（以秒为单位），用于测量短时间的性能。
from time import perf_counter
# 从 tqdm.auto 模块导入 tqdm。
# tqdm 是一个非常流行的库，可以为循环和长时间运行的任务添加一个智能的进度条。
from tqdm.auto import tqdm
# 从 transformers 库导入 AutoTokenizer，用于根据模型名称或路径自动加载对应的分词器。
from transformers import AutoTokenizer
# 导入 torch.multiprocessing 模块，并简写为 mp。
# 这是 PyTorch 提供的多进程库，它对 Python 原生的 multiprocessing 进行了一些封装，以更好地处理 CUDA 张量。
import torch.multiprocessing as mp

# 导入项目内部的模块和类。
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    LLMEngine 是整个推理服务的核心控制器。
    它负责初始化所有组件（配置、模型、分词器、调度器），管理多GPU工作进程，
    并提供一个高级接口来接收请求、驱动推理循环和返回结果。
    """

    def __init__(self, model, **kwargs):
        """
        LLMEngine 的构造函数。
        
        Args:
            model (str): 预训练模型的本地路径。
            **kwargs: 一个可变的关键字参数字典。这是一种非常灵活的Python语法，允许调用者传入任意数量的命名参数。
                      例如 `LLMEngine("path/to/model", max_num_seqs=128, tensor_parallel_size=2)`。
        """
        # --- 1. 初始化配置 (Config) ---
        # 获取 Config dataclass 中定义的所有字段名称，并放入一个集合中以便快速查找。
        config_fields = {field.name for field in fields(Config)}
        # 从传入的 **kwargs 中，只筛选出那些在 Config 类中定义过的参数。
        # Python 语法：这是一个字典推导式（dictionary comprehension）。
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 创建 Config 实例。将模型路径和筛选后的配置参数传入。
        config = Config(model, **config_kwargs)

        # --- 2. 初始化多进程模型执行器 (ModelRunner) ---
        self.ps = []  # 用于存储子进程对象的列表。
        self.events = []  # 用于在主进程和子进程之间同步的事件对象列表。
        
        # 获取一个使用 "spawn" 方法的 multiprocessing 上下文。
        # "spawn" 方法会创建一个全新的 Python 解释器进程，而不是 "fork"（仅在Unix/Linux可用）那样复制父进程。
        # "spawn" 更慢但更安全，尤其是在使用 CUDA 时，可以避免很多潜在问题。
        ctx = mp.get_context("spawn")
        
        # 如果设置了张量并行（tensor_parallel_size > 1），则为每个额外的GPU创建一个子进程。
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 创建一个事件对象，用于子进程通知主进程它已准备就绪。
            # 创建一个进程对象。
            # target=ModelRunner 指定了子进程要执行的函数（或可调用对象，这里是类的构造和运行）。
            # args=(...) 传递给 target 的参数。
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()  # 启动子进程。
            self.ps.append(process)
            self.events.append(event)
        
        # 在主进程中创建第 0 个 ModelRunner 实例（rank 0）。
        # 它会等待所有子进程通过 event 发出“准备就绪”的信号。
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # --- 3. 初始化其他组件 ---
        # 加载分词器。use_fast=True 会尽可能加载一个用 Rust 实现的高性能版本的分词器。
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 从分词器配置中获取结束符的ID，并更新到 config 对象中。
        config.eos = self.tokenizer.eos_token_id
        # 创建调度器实例。
        self.scheduler = Scheduler(config)
        
        # --- 4. 注册清理函数 ---
        # 使用 atexit 注册 self.exit 方法。这确保了当程序退出时，self.exit 会被调用，从而优雅地关闭所有子进程。
        atexit.register(self.exit)

    def exit(self):
        """
        清理和退出函数。
        """
        # 通知所有 ModelRunner 进程退出。
        self.model_runner.call("exit")
        # 删除主进程中的 model_runner 实例，这会触发其 __del__ 方法，释放GPU资源。
        del self.model_runner
        # 等待所有子进程执行完毕并终止。
        # p.join() 会阻塞主进程，直到子进程 p 结束。
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        向引擎添加一个新的生成请求。
        
        Args:
            prompt (str | list[int]): 输入的提示，可以是一个字符串，也可以是已经编码好的 token ID 列表。
            sampling_params (SamplingParams): 这次请求的采样参数（如温度、top_p等）。
        """
        # 检查 prompt 是否为字符串。isinstance 是 Python 中用于检查对象类型的标准方法。
        if isinstance(prompt, str):
            # 如果是字符串，使用分词器将其编码为 token ID 列表。
            prompt = self.tokenizer.encode(prompt)
        # 使用 token ID 列表和采样参数创建一个 Sequence 对象。
        seq = Sequence(prompt, sampling_params)
        # 将创建的序列添加到调度器中，等待被调度。
        self.scheduler.add(seq)

    def step(self):
        """
        执行一个推理步骤。这是整个引擎的核心驱动循环。
        """
        # 1. 调度：从调度器获取一个批次的序列进行处理。
        #    is_prefill 标志表明这个批次是处理提示（prefill）还是生成新token（decode）。
        seqs, is_prefill = self.scheduler.schedule()
        
        # 2. 执行：调用 model_runner 在GPU上运行模型，并获取生成的下一个 token 的 ID。
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        
        # 3. 后处理：使用生成的 token 更新序列的状态，并检查哪些序列已经完成。
        self.scheduler.postprocess(seqs, token_ids)
        
        # 4. 收集结果：从已完成的序列中提取出它们的ID和完整的生成结果。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        
        # 5. 性能统计：计算这个步骤处理的 token 数量，用于计算吞吐量。
        #    如果是 prefill，则为正数；如果是 decode，则为负数（一种区分的技巧）。
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        
        return outputs, num_tokens

    def is_finished(self):
        """
        检查是否所有请求都已处理完毕。
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        一个高级的、端到端的生成接口。
        它接收一批提示，并在内部循环调用 step() 直到所有生成任务完成。
        
        Args:
            prompts: 提示列表。
            sampling_params: 采样参数。可以是一个，也可以是与提示一一对应的列表。
            use_tqdm: 是否显示进度条。

        Returns:
            list[str]: 生成的文本结果列表。
        """
        if use_tqdm:
            # 初始化进度条。
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        # 如果只提供了一个 sampling_params，则为每个 prompt 复制一份。
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # 将所有请求添加到引擎中。
        # zip() 是一个非常有用的内置函数，可以将多个列表打包成一个元组的迭代器。
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
            
        outputs = {}  # 用于存储最终结果的字典，键是序列ID，值是生成的token ID。
        prefill_throughput = decode_throughput = 0.
        
        # 主循环：只要还有未完成的序列，就继续执行推理步骤。
        while not self.is_finished():
            t = perf_counter()  # 记录步骤开始时间。
            output, num_tokens = self.step()
            
            if use_tqdm:
                # 更新性能统计数据。
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                # 在进度条上显示实时吞吐量。
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # 处理刚完成的序列。
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)  # 更新进度条。
                    
        # 按序列ID排序，以确保输出顺序与输入顺序一致。
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 将 token ID 解码回文本字符串。
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        
        if use_tqdm:
            pbar.close()  # 关闭进度条。
            
        return outputs

