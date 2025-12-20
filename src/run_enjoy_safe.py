import torch
import numpy as np
import numpy.dtypes
import sys

# --- 1. 注入白名单 ---
torch.serialization.add_safe_globals([
    np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType, np.dtypes.Int64DType
])

# --- 2. 关键修复：移除之前的 Hacky Patch，因为问题不在 Key 名字，而在模型架构 ---
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# --- 3. 导入 Sample Factory 和 项目模块 ---
from sf_examples.vizdoom.enjoy_vizdoom import main
import src.envs  # 注册环境
from src.models import register_models # <--- 【新增】导入模型注册函数

if __name__ == "__main__":
    # 【新增】必须在 main() 之前注册自定义模型！
    register_models() 
    
    main()
