# 这个 __init__.py 文件有两个主要作用：
# 1. 它将 nanovllm 文件夹标记为一个 Python 包（package），这样您就可以从其他地方导入它。
# 2. 它定义了当外部代码执行 `from nanovllm import ...` 时，可以直接访问的公共API。

# 从 .llm 模块（即 nanovllm/llm.py 文件）中导入 LLM 类。
# 这意味着用户可以直接写 `from nanovllm import LLM`，而不需要写 `from nanovllm.llm import LLM`。
# 这是一种常见的实践，可以提供一个更简洁、更稳定的API接口。
from nanovllm.llm import LLM

# 从 .sampling_params 模块（即 nanovllm/sampling_params.py 文件）中导入 SamplingParams 类。
# 同样，这使得用户可以直接通过 `from nanovllm import SamplingParams` 来使用这个类。
from nanovllm.sampling_params import SamplingParams

