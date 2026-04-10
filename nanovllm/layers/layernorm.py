# 导入 torch 库
import torch
# 从 torch 模块导入 nn
from torch import nn


class RMSNorm(nn.Module):
    """
    实现了 RMSNorm (Root Mean Square Normalization)。
    RMSNorm 是 LayerNorm 的一种变体，它移除了均值中心化步骤，只通过均方根（Root Mean Square）进行缩放。
    它在保持性能的同时，计算上比标准的 LayerNorm 更高效。
    公式为：y = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(
        self,
        hidden_size: int,  # 隐藏层的大小
        eps: float = 1e-6, # 一个很小的数，加在分母上以防止除以零
    ) -> None:
        super().__init__()
        self.eps = eps
        # RMSNorm 只有一个可学习的参数，即缩放因子 weight。
        # 它被初始化为全1。
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # 使用 PyTorch 2.0 的编译器进行优化
    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行标准的 RMSNorm。
        """
        orig_dtype = x.dtype  # 保存原始的数据类型（如 float16）。
        
        # 为了计算精度，将输入 x 转换为 float32。
        x = x.float()
        
        # 计算均方根的平方部分：var = mean(x^2)。
        # pow(2) 计算元素的平方。
        # mean(dim=-1, keepdim=True) 沿着最后一个维度计算均值，并保持该维度大小为1，以便进行广播。
        var = x.pow(2).mean(dim=-1, keepdim=True)
        
        # 计算 1 / sqrt(var + eps)，并原地乘以 x。
        # torch.rsqrt(t) 是计算 t 的平方根倒数（1/sqrt(t)）的高效方法。
        # .mul_() 是原地乘法操作。
        x.mul_(torch.rsqrt(var + self.eps))
        
        # 将数据类型转换回原始类型，并乘以可学习的权重。
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    # 再次使用编译器优化
    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        执行带有残差连接（Residual Connection）的 RMSNorm。
        这种 "Add & Norm" 的模式在 Transformer 中非常常见，将它们融合到一个操作中可以提高效率。
        它先计算 x + residual，然后再进行归一化。
        """
        orig_dtype = x.dtype
        
        # 先将 x 和 residual 都转换为 float32，然后相加。
        x = x.float().add_(residual.float())
        
        # 将相加后的结果（现在是新的残差输入）保存下来，用于下一个模块。
        residual = x.to(orig_dtype)
        
        # 后面的步骤与 rms_forward 完全相同。
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        
        # 返回归一化后的输出和新的残差。
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        模块的主前向传播函数。
        它根据是否提供了 `residual` 输入来决定调用哪个内部方法。
        
        Args:
            x (torch.Tensor): 主输入张量。
            residual (torch.Tensor | None, optional): 残差输入。如果为 None，则执行标准 RMSNorm。
                                                     否则，执行 "Add & Norm"。

        Returns:
            如果 residual is None，返回归一化后的张量。
            如果提供了 residual，返回一个元组 (归一化后的张量, 新的残差)。
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)

