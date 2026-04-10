# 从 dataclasses 模块导入 dataclass 装饰器，用于快速创建数据类。
from dataclasses import dataclass
# 导入 torch 库。
import torch


# 使用 @dataclass 装饰器定义一个上下文类。
# 这个类的作用是作为一个全局的、线程不安全的数据容器，用于在 ModelRunner 的不同方法之间
# （如 prepare_prefill/prepare_decode 和 run_model）以及模型内部的注意力层之间传递当前批次所需的各种元数据。
# 这种使用全局变量的方式简化了函数签名，避免了在每一层都传递大量参数，但在多线程环境下需要特别小心。
@dataclass
class Context:
    """
    Context 类存储了单次模型前向传播所需的上下文信息。
    这些信息对于 FlashAttention 和 PagedAttention 的正确执行至关重要。
    """
    # --- 通用字段 ---
    is_prefill: bool = False  # 标志位，True 表示当前是 prefill 阶段，False 表示是 decode 阶段。

    # --- Prefill 阶段专用字段 ---
    # cu_seqlens_q: 累积序列长度 (Query)。一个一维张量，例如 [0, len_q1, len_q1+len_q2, ...]。
    # FlashAttention 用它来识别批次中每个序列的边界。
    cu_seqlens_q: torch.Tensor | None = None
    # cu_seqlens_k: 累积序列长度 (Key)。在有前缀缓存时，Key 的长度可能与 Query 不同。
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0  # 批次中 Query 的最大长度。
    max_seqlen_k: int = 0  # 批次中 Key 的最大长度。

    # --- 通用字段 (Prefill 和 Decode) ---
    # slot_mapping: 槽位映射。一个一维张量，建立了从逻辑 token（在序列中的位置）到物理 KV 缓存槽位（在巨大张量中的索引）的映射。
    # 这是 PagedAttention 的核心。
    slot_mapping: torch.Tensor | None = None
    # block_tables: 块表。一个二维张量，[num_seqs, max_num_blocks_per_seq]。
    # 每一行代表一个序列，存储了该序列使用的物理块的 ID。
    block_tables: torch.Tensor | None = None
    
    # --- Decode 阶段专用字段 ---
    # context_lens: 上下文长度。一个一维张量，存储批次中每个序列的当前总长度。
    # 在 decode 阶段，注意力计算需要知道每个 token 要关注到多长的历史。
    context_lens: torch.Tensor | None = None

# 创建一个模块级的全局变量 _CONTEXT，它是 Context 类的一个实例。
# 这个变量将在整个程序运行期间存在，并被下面定义的函数所修改和访问。
_CONTEXT = Context()

def get_context():
    """
    获取当前的全局上下文对象。
    模型内部的层（如 Attention）会调用这个函数来获取所需的元数据。
    """
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """
    设置或更新全局上下文对象。
    ModelRunner 在每次执行 `run_model` 之前，会调用这个函数来填充当前批次的上下文信息。
    
    Python 语法：`global _CONTEXT` 声明了我们要修改的是模块级别的全局变量 `_CONTEXT`，
    而不是在函数内部创建一个同名的局部变量。
    """
    global _CONTEXT
    # 创建一个新的 Context 实例并替换掉旧的全局实例。
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """
    重置全局上下文对象，恢复到默认的空状态。
    这个函数在每次 `run` 方法执行完毕后被调用，以清理状态，避免旧的上下文信息泄露到下一个批次。
    """
    global _CONTEXT
    # 创建一个全新的、空的 Context 实例。
    _CONTEXT = Context()

