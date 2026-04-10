# 从 nanovllm.engine.llm_engine 模块导入 LLMEngine 类
from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """
    这是一个高级封装类，直接继承自底层的 LLMEngine。
    
    这种设计的目的是为了提供一个更简洁、更易于用户使用的顶层 API。
    用户可以直接通过 `from nanovllm import LLM` 来实例化和使用模型，
    而不需要关心内部的 `engine` 目录结构。
    
    `pass` 语句表示这个子类没有添加任何新的方法或属性，
    它完全重用了父类 LLMEngine 的所有功能。
    这是一种常见的 API 设计模式，用于简化库的公共接口。
    """
    pass

