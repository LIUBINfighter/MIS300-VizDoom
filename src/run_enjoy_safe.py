import torch
import numpy as np
import numpy.dtypes
import functools

# --- 1. 注入白名单和黑魔法 ---
torch.serialization.add_safe_globals([
    np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType, np.dtypes.Int64DType
])

# 核心手术：拦截 torch.load 并修复 state_dict 的 Key
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # 强制关闭 weights_only 以确保能读取完整字典
    kwargs['weights_only'] = False
    checkpoint = original_torch_load(*args, **kwargs)
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        print("🔧 检测到权重 Key 不匹配，正在进行自动修复...")
        new_model_state = {}
        for k, v in checkpoint["model"].items():
            # 将 encoder.encoders.obs.enc 替换为 encoder.basic_encoder.enc
            new_key = k.replace("encoder.encoders.obs.enc", "encoder.basic_encoder.enc")
            new_model_state[new_key] = v
        checkpoint["model"] = new_model_state
    return checkpoint

torch.load = patched_torch_load

# --- 2. 运行 Enjoy ---
from sf_examples.vizdoom.enjoy_vizdoom import main
import src.envs  # 👈 确保这一行存在，用于注册自定义环境

if __name__ == "__main__":
    # main() 会自动解析 sys.argv
    main()