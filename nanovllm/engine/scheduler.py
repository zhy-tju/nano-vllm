# 从 collections 模块导入 deque，用于实现高效的等待队列和运行队列。
from collections import deque

# 导入项目内部的模块和类。
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    Scheduler 是 vLLM 架构的大脑。它决定了哪些请求（Sequences）在何时以何种方式被执行。
    它负责将等待中的请求转换成可以被模型执行的批次，并管理序列在不同状态（等待、运行、完成）之间的转换。
    其核心目标是最大化 GPU 的利用率。
    """

    def __init__(self, config: Config):
        """
        Scheduler 的构造函数。
        
        Args:
            config (Config): 全局配置对象。
        """
        self.max_num_seqs = config.max_num_seqs  # 一个批次中能处理的最大序列数。
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 一个批次中能处理的最大 token 数。
        self.eos = config.eos  # 结束符的 token ID。
        
        # 创建一个 BlockManager 实例，调度器通过它来管理 KV 缓存块。
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # 使用 deque 创建两个队列：
        # waiting: 存储等待被处理的序列。
        # running: 存储当前正在运行（即已分配了 KV 缓存）的序列。
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """
        检查是否所有的请求都已处理完毕。
        """
        # 当等待队列和运行队列都为空时，说明所有任务都完成了。
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        向调度器中添加一个新的序列。
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        执行一次调度，生成一个要被模型执行的批次。这是调度器的核心方法。
        它实现了 vLLM 的一个关键特性：持续批处理 (Continuous Batching)。
        调度策略优先处理 prefill 请求，然后处理 decode 请求。
        
        Returns:
            tuple[list[Sequence], bool]: 一个元组，包含：
                - list[Sequence]: 被调度执行的序列列表。
                - bool: 一个标志，True 表示这是一个 prefill 批次，False 表示这是一个 decode 批次。
        """
        # --- 阶段一：尝试调度 prefill 请求 ---
        # prefill 指的是处理用户输入的提示（prompt）的阶段。这个阶段计算量大，通常会单独处理。
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        # 只要等待队列不为空，并且当前批次的序列数和 token 数没有超过限制...
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]  # 查看等待队列的第一个序列，但不取出。
            
            # 检查将这个序列加入批次后，是否会超过最大 token 数限制，
            # 以及 BlockManager 是否有足够的空闲块来分配给它。
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break  # 如果不满足条件，则停止添加新的 prefill 请求。
            
            # 满足条件，正式调度这个序列。
            num_seqs += 1
            self.block_manager.allocate(seq)  # 为序列分配 KV 缓存块（这里会利用前缀缓存）。
            num_batched_tokens += len(seq) - seq.num_cached_tokens  # 累加实际需要计算的 token 数量。
            seq.status = SequenceStatus.RUNNING  # 将序列状态更新为 RUNNING。
            
            self.waiting.popleft()  # 从等待队列中移除。
            self.running.append(seq)    # 添加到运行队列中。
            scheduled_seqs.append(seq)
            
        if scheduled_seqs:
            # 如果成功调度了任何 prefill 请求，就立即返回这个 prefill 批次。
            return scheduled_seqs, True

        # --- 阶段二：如果没有任何 prefill 请求，则调度 decode 请求 ---
        # decode 指的是生成新 token 的阶段。这个阶段通常将多个序列的单 token 生成合并到一个批次中。
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()  # 从运行队列的左边取出一个序列进行判断。
            
            # 检查 BlockManager 是否能为这个序列的下一步生成提供空间。
            # 如果不能（例如，所有块都用完了），就需要抢占（preempt）一些序列来腾出空间。
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 优先抢占运行队列中优先级较低的序列（这里简单地抢占队尾的）。
                    self.preempt(self.running.pop())
                else:
                    # 如果运行队列已经没有其他序列可抢占，只能抢占当前序列自己。
                    self.preempt(seq)
                    break  # 跳出内层 while 循环。
            else:
                # Python 语法：`while...else...` 结构。
                # 只有当 while 循环是正常结束（即 `can_append` 返回 True），而不是被 `break` 中断时，才会执行 else 块。
                num_seqs += 1
                self.block_manager.may_append(seq)  # 为下一步生成准备块（可能分配新块或更新哈希）。
                scheduled_seqs.append(seq)
                
        assert scheduled_seqs  # 断言：在 decode 阶段，必须至少调度一个序列，否则引擎会卡住。
        
        # 将调度出的序列重新放回运行队列的头部，以保持队列中序列的原始顺序。
        self.running.extendleft(reversed(scheduled_seqs))
        
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占一个序列。被抢占的序列会从 RUNNING 状态回到 WAITING 状态，
        并释放它所占用的 KV 缓存块，以便其他序列可以使用。
        
        Args:
            seq (Sequence): 要被抢占的序列。
        """
        seq.status = SequenceStatus.WAITING  # 状态改回 WAITING。
        self.block_manager.deallocate(seq)   # 释放其占用的所有块。
        self.waiting.appendleft(seq)         # 将其放回等待队列的头部，以便下次能被优先调度。

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        """
        在模型执行完一个步骤后，进行后处理。
        
        Args:
            seqs (list[Sequence]): 刚刚被执行的序列批次。
            token_ids (list[int]): 模型为每个序列生成的新的 token ID。
        """
        # zip() 函数将两个列表配对遍历。
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)  # 将新生成的 token 添加到序列中。
            
            # 检查序列是否完成。完成的条件是：
            # 1. 生成了结束符（EOS token），并且我们不应该忽略它。
            # 2. 生成的 token 数量达到了用户指定的最大值。
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED  # 状态更新为 FINISHED。
                self.block_manager.deallocate(seq)    # 释放其占用的所有块。
                self.running.remove(seq)              # 从运行队列中彻底移除。

