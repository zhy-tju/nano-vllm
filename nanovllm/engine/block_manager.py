# 从 collections 模块导入 deque。
# deque (double-ended queue) 是一种双端队列，它在列表的两端添加和删除元素都非常快（时间复杂度O(1)），
# 比起 list 在开头添加或删除元素（O(n)）更高效。这里用它来管理空闲块ID，因为我们需要频繁地从一端获取空闲块，并从另一端添加释放的块。
from collections import deque
# 导入 xxhash 模块，这是一个速度极快的非加密哈希算法库。
# 我们用它来为 token 序列块计算哈希值，以实现前缀缓存的快速查找。
import xxhash
# 导入 numpy 库，通常简写为 np。Numpy 是 Python 中用于科学计算的核心库，提供了强大的多维数组对象和相关操作。
# 这里用它将 token ID 列表高效地转换为字节串，以便进行哈希计算。
import numpy as np

# 从 nanovllm.engine.sequence 模块导入 Sequence 类。
# 这表明 BlockManager 需要与 Sequence 对象进行交互。
# Python 语法提示：这里的 `from .sequence import Sequence` 也是合法的，`.` 代表当前目录。
from nanovllm.engine.sequence import Sequence


class Block:
    """
    Block 类代表一个物理的 KV 缓存块。
    它是一个简单的数据结构，用于追踪单个块的状态。
    """

    def __init__(self, block_id):
        """
        Block 类的构造函数。
        
        Args:
            block_id (int): 这个物理块的唯一标识符（ID）。
        """
        self.block_id = block_id  # 物理块的ID
        self.ref_count = 0        # 引用计数：有多少个序列正在使用这个块。当 ref_count 降为 0 时，这个块就可以被回收。
        self.hash = -1            # 存储的 token 序列的哈希值。-1 表示当前没有缓存有效内容，或者内容是动态变化的（例如，正在生成的序列的最后一个块）。
        self.token_ids = []       # 存储在这个块中的 token ID 列表。用于在哈希碰撞时进行精确匹配，确保缓存的正确性。

    def update(self, hash: int, token_ids: list[int]):
        """
        当一个块被用于缓存时，更新它的哈希值和 token ID。
        
        Args:
            hash (int): 新的哈希值。
            token_ids (list[int]): 对应的 token ID 列表。
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        重置一个块的状态，通常在它从空闲状态被分配出去时调用。
        """
        self.ref_count = 1        # 被分配给一个序列，所以引用计数从 1 开始。
        self.hash = -1            # 重置哈希值。
        self.token_ids = []       # 清空 token ID。


class BlockManager:
    """
    BlockManager 是 PagedAttention 机制的核心组件之一。
    它负责管理所有的物理 KV 缓存块（Block），处理块的分配、释放和缓存（共享）。
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        BlockManager 的构造函数。
        
        Args:
            num_blocks (int): 管理的物理块的总数。
            block_size (int): 每个物理块可以存储的 token 数量。
        """
        self.block_size = block_size  # 每个块的大小
        
        # 创建一个包含所有物理块对象的列表。
        # Python 语法：这是一个列表推导式（list comprehension），一种简洁地创建列表的方式。
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        
        # 一个字典，用于存储从哈希值到物理块ID的映射。这是实现前缀缓存（Prefix Caching）的关键。
        self.hash_to_block_id: dict[int, int] = dict()
        
        # 一个双端队列，用于存储所有当前空闲的物理块的ID。
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        
        # 一个集合（set），用于存储所有当前正在被使用的物理块的ID。
        # Python 语法：集合的查找、添加、删除操作的平均时间复杂度是 O(1)，非常高效。
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算一个 token ID 序列的哈希值。这是一个类方法。
        
        Python 语法：@classmethod 装饰器表示这个方法属于类本身，而不是类的实例。
        调用时使用 `BlockManager.compute_hash(...)`。它的第一个参数 `cls` 是类本身，而不是实例 `self`。
        
        Args:
            token_ids (list[int]): 要计算哈希的 token ID 列表。
            prefix (int, optional): 前一个块的哈希值。默认为 -1。
                                    将前一个块的哈希包含进来，可以确保即使两个块的 token 内容相同，
                                    但它们在序列中的位置不同（前缀不同），它们的哈希值也不同。这构成了“链式哈希”。

        Returns:
            int: 计算出的64位哈希值。
        """
        h = xxhash.xxh64()  # 创建一个64位的 xxhash 对象。
        if prefix != -1:
            # 如果提供了前缀哈希，先更新哈希对象。
            # to_bytes(8, "little") 将整数转换为8个字节的字节串。
            h.update(prefix.to_bytes(8, "little"))
        
        # 将 token ID 列表转换为 numpy 数组，再转换为字节串，然后更新哈希。
        # 这样做比直接遍历列表并转换每个整数更高效。
        h.update(np.array(token_ids).tobytes())
        
        # 返回最终的整数哈希摘要。
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        （内部方法）分配一个指定的空闲块。
        Python 约定：以单个下划线 `_` 开头的方法是“内部”或“受保护”的，提示外部调用者不应直接使用它。
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0  # 断言：确保我们分配的块确实是空闲的（引用计数为0）。
        block.reset()  # 重置块的状态。
        self.free_block_ids.remove(block_id)  # 从空闲队列中移除。
        self.used_block_ids.add(block_id)     # 添加到已使用集合中。
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        （内部方法）释放一个指定的块，使其回到空闲池。
        """
        assert self.blocks[block_id].ref_count == 0  # 断言：确保只有在没有序列引用它时才释放。
        self.used_block_ids.remove(block_id)  # 从已使用集合中移除。
        self.free_block_ids.append(block_id)  # 添加回空闲队列的末尾。

    def can_allocate(self, seq: Sequence) -> bool:
        """
        检查是否有足够的空闲块来为一个新的序列分配空间。
        """
        # 比较空闲块的数量和序列所需的块数。
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为一个序列分配物理块。这是实现前缀缓存的核心逻辑。
        """
        assert not seq.block_table  # 断言：确保这个序列之前没有被分配过块。
        h = -1  # 初始化链式哈希的前缀为 -1。
        cache_miss = False  # 标志位，用于追踪是否发生了缓存未命中。
        
        # 遍历序列需要的每一个逻辑块。
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)  # 获取当前逻辑块对应的 token ID 列表。
            
            # 只有当块是满的时候，我们才计算并缓存它。未满的块（通常是序列的最后一个块）是动态的，不缓存。
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            
            # 尝试从缓存中获取块。如果哈希值为 -1（未满的块），则 get 肯定返回 -1。
            block_id = self.hash_to_block_id.get(h, -1)
            
            # 检查缓存是否真的命中。即使哈希匹配，也可能因为哈希碰撞导致 token 内容不同。
            # 所以需要进行二次确认。
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # 发生了缓存未命中。
            
            if cache_miss:
                # 缓存未命中：从空闲队列的开头取一个新块。
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中：
                seq.num_cached_tokens += self.block_size  # 增加序列的缓存 token 计数。
                if block_id in self.used_block_ids:
                    # 如果这个共享的块当前正在被其他序列使用，只需增加它的引用计数。
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 如果这个共享的块存在于缓存中但当前是空闲的（例如，之前使用它的序列都已释放），
                    # 则需要像分配新块一样将其标记为已使用。
                    block = self._allocate_block(block_id)
            
            if h != -1:
                # 如果这个块是可缓存的（即 h != -1），则更新块的信息和哈希表。
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            # 将分配或复用的物理块 ID 添加到序列的 block_table 中。
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        释放一个序列所占用的所有物理块。
        """
        # 反向遍历 block_table。这很重要，但在这个具体实现中不是必须的，不过是一个好习惯。
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1  # 减少块的引用计数。
            if block.ref_count == 0:
                # 如果引用计数降为 0，说明不再有任何序列使用这个块，可以将其释放。
                self._deallocate_block(block_id)
        
        # 重置序列的缓存状态和块表。
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否可以为序列追加一个 token（即执行一步解码）。
        """
        # (len(seq) % self.block_size == 1) 是一个布尔表达式，当序列的长度导致需要一个新块时，它为 True (1)，否则为 False (0)。
        # 所以，这个表达式的意思是：如果追加 token 需要一个新块，我们是否有足够的空闲块（>=1）？
        # 如果不需要新块，我们总能追加（需要 >=0 个空闲块）。
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        为序列的下一步解码准备空间，可能分配一个新块，或者更新最后一个块的哈希。
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        
        # 如果序列的当前长度正好填满了一个或多个块，再追加一个 token 就需要一个新的块。
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1  # 此时，前一个块（现在是倒数第二个）应该是满的，并且已缓存。
            block_id = self.free_block_ids[0]  # 从空闲队列获取一个新块。
            self._allocate_block(block_id)
            block_table.append(block_id)  # 将新块追加到序列的块表中。
        
        # 如果序列追加一个 token 后，正好填满了最后一个块。
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1  # 在填满之前，最后一个块是动态的，未被缓存。
            token_ids = seq.block(seq.num_blocks - 1)  # 获取最后一个块的所有 token。
            
            # 计算链式哈希，其前缀是倒数第二个块的哈希。
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            
            # 现在这个块满了，可以缓存它了。
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 其他情况（即追加后，最后一个块仍未满），最后一个块保持动态，不做任何操作。
            assert last_block.hash == -1

