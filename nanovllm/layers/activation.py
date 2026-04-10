# 导入 torch 库
import torch
# 从 torch 模块导入 nn，这是所有神经网络模块的基类。
from torch import nn
# 从 torch.nn 模块导入 functional，通常简写为 F。
# functional 包含了大量无状态的函数式操作，如激活函数 (relu, silu)、池化等。
# 与 nn.Module 中的层（如 nn.ReLU, nn.SiLU）不同，这些函数不包含可学习的参数。
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    这是一个自定义的神经网络模块，它实现了 "SiLU and Multiply" 操作。
    这个操作是现代 Transformer 模型（如 Llama, Qwen）中前馈网络（FFN）层的一个常见部分，
    通常被称为 SwiGLU。

    它接收一个张量，将其在最后一个维度上分成两半，对第一半应用 SiLU 激活函数，
    然后将结果与第二半逐元素相乘。
    """

    def __init__(self):
        """
        构造函数。
        Python 语法：`super().__init__()` 是一个强制性步骤，用于调用父类（这里是 nn.Module）的构造函数。
        这对于正确初始化模块至关重要。
        """
        super().__init__()

    # @torch.compile 是 PyTorch 2.0 引入的一个强大的 JIT（Just-In-Time）编译器。
    # 它可以将 Python 代码编译成优化的内核（如使用 Triton 语言），从而显著加速计算，
    # 特别是对于多个操作的融合（fusion）。
    # 在这里，它会将下面的 chunk, silu, 和 multiply 操作融合成一个单一的、高效的 GPU 内核，
    # 减少了内存读写和内核启动开销。
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义模块的前向传播逻辑。

        Args:
            x (torch.Tensor): 输入张量。在 FFN 中，这通常是形状为 (..., 2 * hidden_dim) 的张量。

        Returns:
            torch.Tensor: 输出张量，形状为 (..., hidden_dim)。
        """
        # x.chunk(2, -1) 将输入张量 x 在最后一个维度（-1）上分割成 2 个块（chunk）。
        # 例如，如果 x 的形状是 (B, L, 256)，它将被分割成两个形状为 (B, L, 128) 的张量 x 和 y。
        x, y = x.chunk(2, -1)
        
        # F.silu(x) 对第一半张量 x 应用 SiLU (Sigmoid Linear Unit) 激活函数。
        # SiLU(x) = x * sigmoid(x)。
        # 然后，将 SiLU 的结果与第二半张量 y 逐元素相乘。
        return F.silu(x) * y

