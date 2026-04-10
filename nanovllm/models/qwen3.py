# 导入 torch 库
import torch
# 从 torch 模块导入 nn
from torch import nn
# 导入 torch.distributed 模块，简写为 dist，用于分布式计算
import torch.distributed as dist
# 从 transformers 库导入 Qwen3Config，用于加载模型配置
from transformers import Qwen3Config

# 从项目内部的 layers 目录导入所需的模块
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """Qwen3 模型的注意力模块"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()  # 获取张量并行的 world size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0, "头的总数必须能被张量并行大小整除"
        self.num_heads = self.total_num_heads // tp_size  # 每个 GPU 上的 Q 头数量
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0, "KV头的总数必须能被张量并行大小整除"
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # 每个 GPU 上的 KV 头数量
        self.head_dim = head_dim or hidden_size // self.total_num_heads  # 每个头的维度
        self.q_size = self.num_heads * self.head_dim  # 每个 GPU 上 Q 的总维度
        self.kv_size = self.num_kv_heads * self.head_dim  # 每个 GPU 上 K 和 V 的总维度
        self.scaling = self.head_dim ** -0.5  # attention score 的缩放因子
        self.qkv_bias = qkv_bias  # QKV 投影是否使用偏置

        # QKV 投影层，使用列并行
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 输出投影层，使用行并行
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # 获取旋转位置编码 (RoPE)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        # 底层的 Attention 实现
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # 如果 QKV 投影没有偏置，则对 Q 和 K 应用 RMSNorm
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 1. QKV 投影
        qkv = self.qkv_proj(hidden_states)
        # 2. 切分 Q, K, V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 3. Reshape Q, K, V 以匹配头的维度
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 4. 如果没有偏置，应用 LayerNorm
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 5. 应用旋转位置编码
        q, k = self.rotary_emb(positions, q, k)
        # 6. 计算注意力输出
        o = self.attn(q, k, v)
        # 7. 输出投影
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """Qwen3 模型的 MLP/FFN 模块"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # gate_proj 和 up_proj 合并为一个大的列并行线性层
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # [gate_proj_size, up_proj_size]
            bias=False,
        )
        # 下采样投影层，使用行并行
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu", "只支持 silu 激活函数"
        # 使用自定义的 SiluAndMul 激活函数，它将 gate 和 up 的结果相乘
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # 1. 计算 gate 和 up 的投影
        gate_up = self.gate_up_proj(x)
        # 2. 应用 SiLU 激活函数并逐元素相乘
        x = self.act_fn(gate_up)
        # 3. 下采样投影
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 模型的一个 Transformer 解码器层"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 自注意力模块
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        # MLP 模块
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 注意力前的 LayerNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # MLP 前的 LayerNorm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 预归一化 (Pre-Normalization) 结构
        # 1. 应用输入 LayerNorm
        if residual is None:
            # 第一次迭代，residual 就是 hidden_states
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续迭代，将 hidden_states 与上一个残差连接相加后再进行归一化
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # 2. 自注意力模块
        hidden_states = self.self_attn(positions, hidden_states)
        
        # 3. 应用注意力后的 LayerNorm（同样是预归一化）
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # 4. MLP 模块
        hidden_states = self.mlp(hidden_states)
        
        # 返回当前层的输出和残差，用于下一层的输入
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3 核心模型，不包括 LM Head"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 词嵌入层，使用词汇并行
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # 堆叠的解码器层
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 最终的 LayerNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 1. 获取词嵌入
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        # 2. 逐层通过解码器
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 3. 应用最终的 LayerNorm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """完整的 Qwen3 因果语言模型，包含 LM Head"""
    
    # 这是一个映射，用于将 Hugging Face 模型的权重名称映射到我们自定义模型的权重名称
    # 例如，Hugging Face 的 'q_proj' 权重应该加载到我们模型的 'qkv_proj' 参数的 'q' 部分
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        # LM Head，用于将 hidden_states 映射到词汇表大小，使用并行实现
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 如果配置要求权重绑定（tie_word_embeddings），则将 lm_head 的权重与词嵌入的权重共享
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 前向传播，只计算到最后一层的 hidden_states
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 计算最终的 logits
        return self.lm_head(hidden_states)

