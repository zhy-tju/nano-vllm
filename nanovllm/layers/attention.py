# 导入 torch 库
import torch
# 从 torch 模块导入 nn
from torch import nn
# 导入 triton 库。Triton 是 OpenAI 开发的一种 Python-like 语言，用于编写高效的 GPU 内核。
# 它比直接写 CUDA C++ 更简单，同时能生成接近硬件性能极限的代码。
import triton
# 从 triton 库导入 language 模块，通常简写为 tl。它包含了 Triton 语言的核心构建块。
import triton.language as tl

# 从 flash_attn 库导入两个核心函数。
# flash_attn 是一个革命性的库，它实现了一种对内存（IO）感知的精确注意力算法，
# 避免了实例化巨大的 N*N 注意力矩阵，从而极大地加速了计算并减少了显存占用。
# flash_attn_varlen_func: 用于处理一批长度可变的序列（prefill 阶段）。
# flash_attn_with_kvcache: 用于处理自回归解码（decode 阶段），它会直接从提供的 KV 缓存中读取 Key 和 Value。
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
# 从项目工具中导入 get_context 函数，用于获取全局上下文信息。
from nanovllm.utils.context import get_context


# @triton.jit 是一个 JIT (Just-In-Time) 编译器装饰器，它会将下面的 Python 函数编译成一个 GPU 内核。
@triton.jit
def store_kvcache_kernel(
    key_ptr,            # 指向输入 Key 张量的指针
    key_stride,         # Key 张量在第0维上的步长（stride）
    value_ptr,          # 指向输入 Value 张量的指针
    value_stride,       # Value 张量在第0维上的步长
    k_cache_ptr,        # 指向物理 K 缓存张量的指针
    v_cache_ptr,        # 指向物理 V 缓存张量的指针
    slot_mapping_ptr,   # 指向槽位映射（slot_mapping）张量的指针
    D: tl.constexpr,    # 每个 token 的 K/V 数据大小（head_dim * num_heads），作为编译时常量
):
    """
    这是一个 Triton 内核，它的功能是将计算出的 Key 和 Value 写入到 PagedAttention 的物理缓存中。
    这个内核是并行执行的，每个程序实例（program instance）处理一个 token。
    """
    # tl.program_id(0) 获取当前程序实例的唯一 ID，范围是 [0, N-1]，其中 N 是启动内核时指定的程序数量。
    # 在这里，它对应于批次中 token 的索引。
    idx = tl.program_id(0)
    
    # 从 slot_mapping 中加载当前 token 应该被写入的物理槽位（slot）的地址。
    slot = tl.load(slot_mapping_ptr + idx)
    
    # 如果 slot 为 -1，表示这个 token 不需要被缓存（例如，在某些填充或特殊情况下），直接返回。
    if slot == -1: return
    
    # --- 加载输入的 K 和 V ---
    # tl.arange(0, D) 创建一个从 0 到 D-1 的向量。
    # 计算当前 token 的 Key 数据在输入张量中的所有偏移量。
    key_offsets = idx * key_stride + tl.arange(0, D)
    # 计算 Value 的偏移量。
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 从内存中加载完整的 Key 向量。
    key = tl.load(key_ptr + key_offsets)
    # 加载 Value 向量。
    value = tl.load(value_ptr + value_offsets)
    
    # --- 写入到 KV 缓存 ---
    # 计算当前 token 在物理缓存张量中的所有偏移量。
    cache_offsets = slot * D + tl.arange(0, D)
    # 将 Key 向量存储到 K 缓存的指定位置。
    tl.store(k_cache_ptr + cache_offsets, key)
    # 将 Value 向量存储到 V 缓存的指定位置。
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    这是一个 Python 函数，用于启动上面的 Triton 内核。
    它负责准备参数并以正确的配置启动内核。
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 断言检查，确保张量是连续的，以便 Triton 内核可以正确地进行内存访问。
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    
    # 启动内核。
    # `[(N,)]` 定义了网格（grid）的维度，这里表示我们要启动 N 个并行的程序实例，每个实例处理一个 token。
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """
    自定义的注意力模块，它封装了 PagedAttention 的核心逻辑。
    它根据当前的推理阶段（prefill 或 decode）选择合适的 FlashAttention 函数，
    并负责调用 Triton 内核将计算出的 K/V 写入缓存。
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 初始化 k_cache 和 v_cache 为空张量。
        # 它们将在 ModelRunner 的 allocate_kv_cache 方法中被替换为指向真实物理缓存的切片。
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        前向传播函数。
        
        Args:
            q (torch.Tensor): Query 张量。
            k (torch.Tensor): Key 张量。
            v (torch.Tensor): Value 张量。
        """
        context = get_context()  # 获取全局上下文。
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # 如果物理缓存已分配，并且有需要写入的槽位...
        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            # 调用 Triton 内核将当前计算出的 k 和 v 写入到物理缓存中。
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            # --- Prefill 阶段 ---
            if context.block_tables is not None:
                # 如果存在前缀缓存（通过 block_tables 判断），
                # 那么注意力计算的 K 和 V 应该直接使用完整的物理缓存，而不是刚计算出的那一小部分 k, v。
                k, v = k_cache, v_cache
            
            # 调用 FlashAttention 的变长序列版本。
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables  # 传入块表以支持 PagedAttention
            )
        else:
            # --- Decode 阶段 ---
            # 调用 FlashAttention 的 KV 缓存版本。
            # q.unsqueeze(1) 将 Query 的形状从 [B, D] 变为 [B, 1, D]，因为 decode 阶段每个序列只处理一个 token。
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,  # 直接传入完整的 K 缓存
                v_cache,  # 直接传入完整的 V 缓存
                cache_seqlens=context.context_lens,  # 告诉 FlashAttention 每个序列的真实长度
                block_table=context.block_tables,    # 传入块表
                softmax_scale=self.scale,
                causal=True
            )
        return o

