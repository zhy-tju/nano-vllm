# 导入 torch 库
import torch
# 从 torch 模块导入 nn
from torch import nn
# 从 torch.nn 模块导入 functional，简写为 F
import torch.nn.functional as F
# 导入 torch.distributed 模块，简写为 dist，用于分布式计算
import torch.distributed as dist


def divide(numerator, denominator):
    """一个辅助函数，确保除法可以整除。"""
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """
    所有自定义并行线性层的基类。
    它处理了张量并行（Tensor Parallelism, TP）相关的通用初始化逻辑。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,  # 张量并行的切分维度，0表示按行切（列并行），1表示按列切（行并行）
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # 创建权重参数。注意，这里的 output_size 和 input_size 可能已经是被切分后的大小。
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 附加一个默认的权重加载器，子类可以覆盖它。
        self.weight.weight_loader = self.weight_loader
        
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            # 即使没有偏置，也显式地将 self.bias 注册为 None。
            # 这是一种良好的实践，可以避免在 forward 中出现属性错误。
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基类不实现 forward 方法，强制子类必须实现自己的 forward 逻辑。
        raise NotImplementedError

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 基类的权重加载器，同样需要子类覆盖。
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    一个普通的、非并行的线性层。
    它的权重在所有 GPU 上都是完全复制的（Replicated）。
    """
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 直接将加载的权重完整地复制到参数中。
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 执行标准的线性变换。
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层 (Column Parallel Linear)。
    权重矩阵 W 被沿着列（output_size 维度）切分成 N 份，每个 GPU 持有一份 W_i。
    输入 x 在所有 GPU 上是相同的。
    每个 GPU 计算 Y_i = x @ W_i.T。
    计算结果 Y = [Y_0, Y_1, ..., Y_{N-1}] 是一个切分后的张量，需要后续的 all-gather 或其他操作来合并。
    """
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size()
        # 调用基类构造函数。注意 output_size 被切分了。
        # tp_dim=0 表示权重矩阵 self.weight (shape: [output_size, input_size]) 是在第0维上被切分的。
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # 获取当前分片在切分维度上的大小。
        shard_size = param_data.size(self.tp_dim)
        # 计算当前 GPU 应该加载的分片的起始索引。
        start_idx = self.tp_rank * shard_size
        # 从完整的权重中切出对应的分片。
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 直接进行线性计算。由于输入 x 是完整的，而权重 self.weight 是分片的，
        # 所以输出也是分片的，不需要额外的通信。
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并的列并行线性层。
    用于处理像 Llama 中的 FFN 层，其中 gate_proj 和 up_proj 被合并成一个大的线性层来计算，
    但它们的权重在物理上是分开存储的。
    这个类负责将这些分开存储的权重加载到正确的位置。
    """
    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        self.output_sizes = output_sizes
        # 总的输出大小是所有部分的总和。
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        # 计算当前加载的权重分片（如 gate_proj 或 up_proj）在合并后的大参数张量中的偏移量。
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        # 计算这个分片的大小。
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 切出参数张量中对应的区域。
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从加载的权重中，根据当前 GPU 的 rank 切出对应的列并行分片。
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    专门为 Q, K, V 的计算设计的并行线性层。
    通常，Q, K, V 的权重在存储时被合并成一个大的 qkv_proj 权重。
    这个类负责将这个大的权重加载进来，并正确地切分给每个 GPU。
    """
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        # 计算每个 GPU 负责的头数量。
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 计算总的输出大小。
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        # 根据当前加载的是 Q, K, 还是 V，计算它在合并后的 QKV 参数张量中的偏移量和大小。
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else: # "v"
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        
        # 切出参数张量中对应的区域。
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从加载的权重中，根据当前 GPU 的 rank 切出对应的列并行分片。
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    行并行线性层 (Row Parallel Linear)。
    权重矩阵 W 被沿着行（input_size 维度）切分成 N 份，每个 GPU 持有一份 W_i。
    输入 x 也被认为是沿着特征维度切分的，每个 GPU 持有 x_i。
    每个 GPU 计算 Y_i = x_i @ W_i.T。
    最终的输出 Y = sum(Y_i)，这通过一个 all_reduce 操作完成。
    """
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size()
        # 调用基类构造函数。注意 input_size 被切分了。
        # tp_dim=1 表示权重矩阵 self.weight (shape: [output_size, input_size]) 是在第1维上被切分的。
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 已经是被切分的了。
        # 每个 GPU 计算出自己的部分结果 y。
        # 只有 rank 0 需要加上偏置项，否则偏置会被加 N 次。
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            # 使用 all_reduce 将所有 GPU 上的部分结果 y 相加。
            # 操作完成后，每个 GPU 上的 y 都包含了最终的、完整的输出结果。
            dist.all_reduce(y)
        return y

