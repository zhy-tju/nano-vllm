# 导入 os 模块，用于处理文件路径。
import os
# 从 glob 模块导入 glob 函数。
# glob 用于查找符合特定规则的文件路径名，支持通配符 `*`, `?`, `[]`。
from glob import glob
# 导入 torch 库。
import torch
# 从 torch 模块导入 nn，这是 PyTorch 中所有神经网络模块的基类。
from torch import nn
# 从 safetensors 库导入 safe_open 函数。
# Safetensors 是一种新的、安全的张量存储格式，它比 pickle（.bin 文件）更安全、更快。
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    一个默认的权重加载函数。
    它的作用是将从文件中加载的权重张量（loaded_weight）复制到模型的参数张量（param）中。
    
    Args:
        param (nn.Parameter): 模型中定义的参数，例如 `self.weight`。
        loaded_weight (torch.Tensor): 从 .safetensors 文件中加载的权重。
    """
    # param.data 指的是参数张量的数据部分，对其进行原地操作（in-place operation）不会被梯度追踪。
    # .copy_() 是一个原地操作，它将 `loaded_weight` 的内容复制到 `param.data` 中。
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    加载预训练模型权重的主函数。
    它会遍历指定路径下的所有 .safetensors 文件，并将权重加载到给定的模型中。
    这个函数还特别处理了被合并的权重（如 QKV 矩阵）和张量并行的情况。

    Args:
        model (nn.Module): 要加载权重的模型实例。
        path (str): 包含 .safetensors 文件的目录路径。
    """
    # 检查模型对象是否有一个名为 "packed_modules_mapping" 的属性。
    # 这个映射用于处理一些特殊的权重加载情况，比如当多个逻辑上的参数（如Q, K, V的权重）
    # 在物理上被打包存储成一个大的张量时。
    # Python 语法：getattr(object, name, default) 会获取对象的 `name` 属性，如果不存在，则返回 `default` 值。
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # 使用 glob 查找路径下所有的 .safetensors 文件。
    # os.path.join(path, "*.safetensors") 会安全地拼接路径，例如 "my/model/" + "*.safetensors" -> "my/model/*.safetensors"。
    for file in glob(os.path.join(path, "*.safetensors")):
        # 使用 safe_open 打开 safetensors 文件。
        # "pt" 表示我们期望加载 PyTorch 张量，"cpu" 表示先将张量加载到 CPU 内存中。
        # Python 语法：`with ... as ...` 是一种上下文管理器，它能确保在代码块执行完毕后，文件被正确关闭，即使发生错误。
        with safe_open(file, "pt", "cpu") as f:
            # 遍历文件中的所有权重名称（key）。
            for weight_name in f.keys():
                # --- 处理被打包的权重 ---
                # 遍历 packed_modules_mapping 中的特殊规则。
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 如果文件中的权重名称包含了特殊规则的键（例如 "qkv_proj"）。
                        v, shard_id = packed_modules_mapping[k]
                        # 将权重名称中的键替换为目标参数名（例如 "qkv_proj" -> "query"）。
                        param_name = weight_name.replace(k, v)
                        # 从模型中获取对应的参数对象。
                        param = model.get_parameter(param_name)
                        # 获取该参数上附加的自定义权重加载器。
                        weight_loader = getattr(param, "weight_loader")
                        # 调用自定义加载器，传入加载的权重和分片ID。
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        # 处理完毕，跳出内层 for 循环。
                        break
                # Python 语法：`for...else...` 结构。
                # 只有当 for 循环正常执行完毕（即没有被 `break` 中断）时，才会执行 else 块。
                else:
                    # --- 处理普通权重 ---
                    # 如果上面的 for 循环没有被 break，说明这不是一个被打包的权重。
                    # 直接使用文件中的权重名称去模型中查找对应的参数。
                    param = model.get_parameter(weight_name)
                    # 尝试获取参数上附加的自定义加载器，如果不存在，则使用我们定义的 `default_weight_loader`。
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    # 调用加载器加载权重。
                    weight_loader(param, f.get_tensor(weight_name))

