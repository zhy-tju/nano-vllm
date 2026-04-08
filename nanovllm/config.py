# 导入 os 模块，用于与操作系统进行交互，如此处用于检查路径是否为目录。
import os
# 从 dataclasses 模块导入 dataclass 装饰器。
# dataclass 可以自动为类添加特殊方法，如 __init__(), __repr__() 等，让创建数据类变得非常简洁。
from dataclasses import dataclass
# 从 transformers 库导入 AutoConfig，这是一个非常方便的类，可以根据模型名称或路径自动加载对应的配置信息。
from transformers import AutoConfig


# @dataclass 是一个装饰器（decorator），它会告诉 Python 自动为下面的 Config 类生成一些样板代码。
# 比如，你不需要自己写 __init__ 构造函数来初始化 self.model = model, self.max_num_seqs = max_num_seqs 等。
# 这让代码更整洁，更专注于数据本身。
@dataclass
class Config:
    """
    这个 Config 类是一个数据容器，用于集中管理 nano-vllm 引擎的所有配置参数。
    使用 dataclass 可以让我们清晰地定义每个配置项及其类型和默认值。
    """
    # model: str
    # 定义一个名为 model 的字段，它的类型是字符串（str）。
    # 这个字段将存储预训练模型在本地的路径。
    model: str
    
    # max_num_batched_tokens: int = 16384
    # 定义一个整型字段，表示一个批次（batch）中能处理的最大 token 数量。默认值为 16384。
    # 这是控制GPU显存占用的一个关键参数。
    max_num_batched_tokens: int = 16384
    
    # max_num_seqs: int = 512
    # 定义一个整型字段，表示引擎能同时处理的最大序列（请求）数量。默认值为 512。
    max_num_seqs: int = 512
    
    # max_model_len: int = 4096
    # 定义一个整型字段，表示模型能处理的单个序列的最大长度。默认值为 4096。
    # 这个值会与模型自身的 `max_position_embeddings` 取较小值。
    max_model_len: int = 4096
    
    # gpu_memory_utilization: float = 0.9
    # 定义一个浮点型字段，表示GPU显存的使用率。默认值为 0.9，即使用90%的显存。
    # 引擎会根据这个值来计算可以分配多少KV缓存。
    gpu_memory_utilization: float = 0.9
    
    # tensor_parallel_size: int = 1
    # 定义一个整型字段，表示张量并行的规模（即使用多少个GPU来切分模型）。默认为1，表示不使用张量并行。
    tensor_parallel_size: int = 1
    
    # enforce_eager: bool = False
    # 定义一个布尔型字段，如果为 True，则强制使用 Eager 模式执行，而不是使用编译优化（如 aot_autograd）。
    # 主要用于调试。默认为 False。
    enforce_eager: bool = False
    
    # hf_config: AutoConfig | None = None
    # 定义一个字段，用于存储从 Hugging Face 加载的模型配置对象。
    # 它的类型是 `AutoConfig` 或者 `None`（Python 3.10+ 的新语法，等价于 Union[AutoConfig, None]）。
    # 默认值为 None，会在 `__post_init__` 中被初始化。
    hf_config: AutoConfig | None = None
    
    # eos: int = -1
    # 定义一个整型字段，存储模型的结束符（End-Of-Sequence）的 token ID。默认为 -1，稍后会自动从模型配置中读取。
    eos: int = -1
    
    # kvcache_block_size: int = 256
    # 定义一个整型字段，表示KV缓存块的大小。这是 PagedAttention 机制的核心参数之一。默认为 256。
    kvcache_block_size: int = 256
    
    # num_kvcache_blocks: int = -1
    # 定义一个整型字段，表示总共要分配多少个KV缓存块。默认为 -1，表示稍后会根据显存使用率自动计算。
    num_kvcache_blocks: int = -1

    # def __post_init__(self):
    # 这是 dataclass 提供的一个特殊方法，在对象通过自动生成的 __init__ 方法初始化完毕后被调用。
    # 我们可以在这里进行一些额外的初始化或验证操作。
    def __post_init__(self):
        # assert 是一种断言语句，用于检查一个条件是否为真。如果条件为假，程序会抛出 AssertionError 异常。
        # 这是一种很好的防御性编程实践，确保传入的参数是有效的。
        
        # 检查 self.model 是否是一个真实存在的目录。
        assert os.path.isdir(self.model)
        
        # 检查KV缓存块大小是否为256的整数倍。这是一个设计上的约束。
        assert self.kvcache_block_size % 256 == 0
        
        # 检查张量并行的大小是否在1到8之间。
        assert 1 <= self.tensor_parallel_size <= 8
        
        # 使用 AutoConfig.from_pretrained 方法从指定的模型路径加载 Hugging Face 的模型配置。
        # 并将其赋值给 self.hf_config 字段。
        self.hf_config = AutoConfig.from_pretrained(self.model)
        
        # 确定模型最终能处理的最大长度。
        # 它不能超过我们在 Config 中设置的 max_model_len，也不能超过模型本身支持的最大长度（hf_config.max_position_embeddings）。
        # 所以我们取这两者中的较小值。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        
        # 确保一个批次能容纳的最大 token 数至少要大于等于模型能处理的单个序列的最大长度。
        # 否则，一个非常长的序列可能永远无法被处理。
        assert self.max_num_batched_tokens >= self.max_model_len

