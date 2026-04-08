# 从 copy 模块导入 copy 函数。
# copy() 用于创建一个对象的浅拷贝。对于列表来说，这意味着创建一个新列表，但列表中的元素是原始元素的引用。
# 这里用它来复制 token_ids，以避免修改传入的原始列表。
from copy import copy
# 从 enum 模块导入 Enum 和 auto。
# Enum 是创建枚举类型的基类。枚举是一组有名字的常量。
# auto() 是一个辅助函数，可以自动为枚举成员分配一个值。
from enum import Enum, auto
# 从 itertools 模块导入 count。
# count() 创建一个迭代器，它会从指定的起始值开始，无限地生成连续的整数。
# 这里用它来为每个新创建的序列生成一个唯一的、自增的ID。
from itertools import count

# 导入项目内部的 SamplingParams 类。
from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    这是一个枚举类，定义了序列（Sequence）可能处于的几种状态。
    使用枚举可以使代码更清晰、更易读，避免使用魔法数字或字符串（如 status="running"）。
    """
    WAITING = auto()   # 序列正在等待被调度。auto() 会自动为其分配一个值（如 1）。
    RUNNING = auto()   # 序列正在被模型处理（已分配KV缓存）。auto() 会分配下一个值（如 2）。
    FINISHED = auto()  # 序列已完成生成。auto() 会分配再下一个值（如 3）。


class Sequence:
    """
    Sequence 类是 vLLM 中对一个生成请求的内部表示。
    它不仅仅包含 token ID，还追踪了该请求的状态、KV缓存信息、采样参数等所有相关信息。
    """
    # 类变量（Class Variable），被该类的所有实例共享。
    block_size = 256  # 静态地存储块大小，方便内部计算。
    counter = count() # 创建一个全局的序列ID计数器。

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        Sequence 类的构造函数。
        
        Args:
            token_ids (list[int]): 输入的提示（prompt）对应的 token ID 列表。
            sampling_params (SamplingParams, optional): 该请求的采样参数。默认为一个默认的 SamplingParams 对象。
        """
        # --- 基本信息 ---
        self.seq_id = next(Sequence.counter)  # 从全局计数器获取一个唯一的序列ID。
        self.status = SequenceStatus.WAITING  # 新创建的序列初始状态为 WAITING。
        self.token_ids = copy(token_ids)      # 存储完整的 token 序列（包括 prompt 和生成的 completion）。
        self.last_token = token_ids[-1]       # 序列中的最后一个 token，在 decode 阶段作为输入。
        self.num_tokens = len(self.token_ids) # 序列的当前总长度。
        self.num_prompt_tokens = len(token_ids) # 提示部分的 token 数量，这个值是固定的。
        
        # --- 缓存和物理布局信息 ---
        self.num_cached_tokens = 0  # 已被计算并存入 KV 缓存的 token 数量（主要用于前缀缓存）。
        self.block_table = []       # 一个列表，存储该序列占用的物理块的 ID。这是逻辑到物理的映射表。
        
        # --- 采样参数 ---
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """
        Python 的特殊方法（dunder method），允许我们对 Sequence 对象使用 `len()` 函数。
        例如 `len(my_sequence)` 会调用这个方法。
        """
        return self.num_tokens

    def __getitem__(self, key):
        """
        特殊方法，允许我们像列表一样通过索引或切片来访问 Sequence 对象的 token。
        例如 `my_sequence[5]` 或 `my_sequence[5:10]`。
        """
        return self.token_ids[key]

    # Python 语法：@property 装饰器可以将一个方法变成一个只读属性。
    # 这样我们就可以像访问变量一样调用它（例如 `seq.is_finished`），而不需要加括号 `seq.is_finished()`。
    # 这让代码更简洁，更符合属性的语义。
    @property
    def is_finished(self):
        """检查序列是否已完成。"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """计算已生成的 token 的数量。"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """获取提示部分的 token ID 列表。"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """获取生成部分的 token ID 列表。"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """计算被前缀缓存的块的数量。"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """计算序列总共占用的逻辑块的数量。"""
        # 这是一个向上取整的常用技巧：(a + b - 1) // b
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """计算最后一个块中包含的 token 数量。"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取第 i 个逻辑块中的 token ID。
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """
        向序列追加一个新生成的 token。
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # __getstate__ 和 __setstate__ 是两个特殊方法，用于控制对象在被 pickle 序列化和反序列化时的行为。
    # 这在多进程通信中非常重要，因为对象需要在进程间传递。
    # 默认的 pickle 行为会序列化整个对象的 __dict__，但在这里我们通过自定义这两个方法，
    # 可以更精细地控制哪些数据被传递，从而优化通信效率。
    def __getstate__(self):
        """
        定义当 `pickle.dumps(self)` 被调用时，应该返回什么。
        这里我们只选择性地返回了几个关键状态，而不是整个对象。
        """
        # 对于已生成的序列，我们不需要传递完整的 token_ids 列表，只需要传递最后一个 token 就够了，
        # 因为其他信息（如 block_table）已经包含了历史。这可以节省大量的进程间通信开销。
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """
        定义当 `pickle.loads(bytes)` 创建新对象时，如何用 `state` 来恢复对象的状态。
        `state` 就是 `__getstate__` 返回的元组。
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            # 如果是 prefill 阶段的序列，恢复完整的 token_ids。
            self.token_ids = state[-1]
        else:
            # 如果是 decode 阶段的序列，只恢复 last_token。
            self.last_token = state[-1]

