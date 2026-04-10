# 从 functools 模块导入 lru_cache，用于缓存函数结果
from functools import lru_cache
# 导入 torch 库
import torch
# 从 torch 模块导入 nn
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,  # 输入张量，通常是 query 或 key
    cos: torch.Tensor,  # 预先计算好的余弦值
    sin: torch.Tensor,  # 预先计算好的正弦值
) -> torch.Tensor:
    """
    应用旋转位置编码。
    这个函数实现了 RoPE 的核心数学变换。
    """
    # 1. 将输入张量 x 在最后一个维度上切分成两半。
    #    x1 和 x2 分别对应于 RoPE 变换公式中的不同部分。
    #    使用 .float() 是为了确保计算精度，避免在半精度（如 bfloat16）下出现问题。
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    
    # 2. 应用旋转变换。
    #    这等价于将 (x1, x2) 视为一个复数 x1 + i*x2，然后乘以另一个复数 cos + i*sin。
    #    展开后就是：(x1 + i*x2) * (cos + i*sin) = (x1*cos - x2*sin) + i*(x2*cos + x1*sin)
    y1 = x1 * cos - x2 * sin  # 变换后的第一部分（实部）
    y2 = x2 * cos + x1 * sin  # 变换后的第二部分（虚部）
    
    # 3. 将变换后的两部分重新拼接起来，并转换回原始的数据类型。
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块。
    这个类负责预计算和缓存 RoPE 所需的 sin 和 cos 值。
    """

    def __init__(
        self,
        head_size: int,  # 注意力头的维度
        rotary_dim: int,  # 旋转编码的维度，通常等于 head_size
        max_position_embeddings: int,  # 模型支持的最大序列长度
        base: float,  # RoPE 中的基数，通常是 10000.0
    ) -> None:
        super().__init__()
        self.head_size = head_size
        # 确保旋转编码的维度等于注意力头的维度
        assert rotary_dim == head_size
        
        # 1. 计算频率 `inv_freq`。这是 RoPE 的核心部分。
        #    公式为 1 / (base^(2k / d))，其中 d 是 rotary_dim，k 是 0, 1, ..., d/2 - 1。
        #    torch.arange(0, rotary_dim, 2, ...) 生成了 0, 2, 4, ..., rotary_dim-2 的序列，代表 2k。
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # 2. 创建位置索引 `t`，从 0 到 max_position_embeddings - 1。
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        
        # 3. 计算频率和位置的乘积 `freqs`。
        #    `einsum` 操作计算了 `t` 和 `inv_freq` 的外积，结果的形状是 [max_position_embeddings, rotary_dim / 2]。
        #    这得到了每个位置上每个频率分量的角度。
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        
        # 4. 计算所有位置的 cos 和 sin 值。
        cos = freqs.cos()
        sin = freqs.sin()
        
        # 5. 将 cos 和 sin 拼接在一起，并增加一个维度用于广播。
        #    最终 cache 的形状是 [max_position_embeddings, 1, rotary_dim]。
        #    unsqueeze_(1) 是一个原地操作，在第1维增加一个维度。
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        
        # 6. 将计算好的 cache 注册为模型的 buffer。
        #    buffer 是模型状态的一部分，但不是模型参数（不会被优化器更新）。
        #    `persistent=False` 表示这个 buffer 不需要被保存到模型的 state_dict 中。
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,  # 输入序列中每个 token 的位置索引
        query: torch.Tensor,      # query 张量
        key: torch.Tensor,        # key 张量
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. 从缓存中根据 `positions` 索引出对应的 sin/cos 值。
        #    `cos_sin_cache` 的形状是 [max_pos, 1, dim]，`positions` 的形状是 [batch_size, seq_len]。
        #    索引后 `cos_sin` 的形状是 [batch_size, seq_len, 1, dim]，可以和 Q, K 进行广播操作。
        cos_sin = self.cos_sin_cache[positions]
        
        # 2. 将 `cos_sin` 张量切分成 cos 和 sin 两部分。
        cos, sin = cos_sin.chunk(2, dim=-1)
        
        # 3. 分别对 query 和 key 应用旋转编码。
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        
        # 4. 返回编码后的 query 和 key。
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    一个带缓存的工厂函数，用于创建 RotaryEmbedding 实例。
    `@lru_cache(1)` 装饰器表示这个函数的结果会被缓存。因为 RoPE 的参数在一次运行中通常是固定的，
    所以多次调用这个函数（使用相同的参数）将直接返回缓存的 `RotaryEmbedding` 实例，而不会重新创建。
    这可以节省初始化时间。
    """
    # 目前还不支持 rope_scaling，如果提供了这个参数，就直接断言失败。
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb

