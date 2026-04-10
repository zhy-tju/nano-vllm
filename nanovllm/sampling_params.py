# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    一个数据类 (dataclass)，用于封装文本生成过程中的采样参数。
    
    `@dataclass` 装饰器会自动为这个类生成一些特殊方法，
    比如 `__init__`、`__repr__`、`__eq__` 等，
    使得创建和使用这种主要用于存储数据的类变得非常方便。
    """
    
    # 温度 (temperature): 控制生成文本的随机性。
    # - 较高的温度（如 1.0）会产生更多样、更随机的文本。
    # - 较低的温度（如 0.1）会使模型更倾向于选择高概率的词，生成的文本更确定、更保守。
    # 默认值为 1.0。
    temperature: float = 1.0
    
    # 最大生成 token 数 (max_tokens): 指定模型在停止前最多可以生成的 token 数量。
    # 这是一个重要的停止条件，可以防止无限生成。
    # 默认值为 64。
    max_tokens: int = 64
    
    # 是否忽略 EOS (End-of-Sequence) token:
    # 如果为 True，即使模型生成了 EOS token，也不会停止生成，直到达到 max_tokens。
    # 如果为 False，一旦生成 EOS token，生成过程就会立即停止。
    # 默认值为 False。
    ignore_eos: bool = False

    def __post_init__(self):
        """
        这是 dataclass 提供的一个特殊方法，在 `__init__` 方法被调用之后自动执行。
        我们用它来对参数进行验证。
        """
        # 断言温度值必须大于一个很小的正数 (1e-10)。
        # 这意味着完全的贪心采样（temperature=0）是不被允许的。
        # 这是因为项目中的采样器实现（Gumbel-Max trick）在温度为0时可能会遇到数值问题。
        assert self.temperature > 1e-10, "greedy sampling is not permitted"

