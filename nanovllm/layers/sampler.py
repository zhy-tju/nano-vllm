# 导入 torch 库
import torch
# 从 torch 模块导入 nn
from torch import nn


class Sampler(nn.Module):
    """
    采样器模块。
    负责从模型输出的 logits 中采样生成下一个 token。
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        执行采样操作。
        
        Args:
            logits (torch.Tensor): 模型的原始输出，形状为 [batch_size, vocab_size]。
                                   表示每个词汇表中 token 的未归一化分数。
            temperatures (torch.Tensor): 温度参数，形状为 [batch_size]。
                                         每个序列可以有不同的温度。
        
        Returns:
            torch.Tensor: 采样得到的 token ID，形状为 [batch_size]。
        """
        
        # 1. 应用温度缩放 (Temperature Scaling)
        #    - `logits.float()`: 将 logits 转换为浮点数以保证计算精度。
        #    - `temperatures.unsqueeze(dim=1)`: 将温度张量从 [batch_size] 扩展为 [batch_size, 1]，
        #      以便与 [batch_size, vocab_size] 的 logits 进行广播除法。
        #    - `div_()`: 原地除法操作。每个 token 的 logit 都会被其对应序列的温度值所除。
        #      - 温度 > 1: 使得概率分布更平滑，增加采样多样性（更随机）。
        #      - 温度 < 1: 使得概率分布更尖锐，让模型更倾向于选择高概率的 token（更确定）。
        #      - 温度 = 0: 会导致除以零，这里没有处理，但通常意味着贪心采样（argmax）。
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        
        # 2. 计算概率分布
        #    - `torch.softmax(logits, dim=-1)`: 对缩放后的 logits 应用 softmax 函数，
        #      将其转换为一个合法的概率分布，其中所有 token 的概率之和为 1。
        probs = torch.softmax(logits, dim=-1)
        
        # 3. Gumbel-Max 技巧进行采样
        #    这是一种从分类分布中进行可微分采样的高效方法，等价于从 `probs` 定义的多项式分布中采样。
        #    - `torch.empty_like(probs).exponential_(1)`: 生成一个与 `probs` 形状相同、
        #      服从指数分布（lambda=1）的随机噪声张量 G。
        #    - `.clamp_min_(1e-10)`: 确保噪声值不为零，避免后续除法出现问题。
        #    - `probs.div_(...)`: 计算 `log(probs) - G` 的一种数值稳定方式的变体。
        #      这里直接用 `probs / G`，其中 G 是指数分布随机数。
        #      这可以看作是 Gumbel-Max 技巧的一种实现。
        #    - `.argmax(dim=-1)`: 找到具有最大 `probs / G` 值的 token 的索引。
        #      这个索引就是我们最终采样得到的 token。
        #      这种方法避免了直接使用 `torch.multinomial`，在某些硬件上可能更高效。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        
        return sample_tokens

