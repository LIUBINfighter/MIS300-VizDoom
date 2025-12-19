# PyTorch 2.6+ 与 Sample Factory 兼容性修复手册

在本项目中，我们遇到了 PyTorch 2.6+ 引入的安全加载机制（PEP 715）以及 Sample Factory 内部结构变动带来的挑战。以下是解决方案的详细记录。

## 1. UnpicklingError (安全全局变量)

### 问题描述
尝试加载包含 Numpy 标量或特定 DType 的 Checkpoint 时，PyTorch 会报错：
`Weights only load failed. Re-running with weights_only=False ... UnpicklingError: Global 'numpy.core.multiarray.scalar' is not allowed.`

### 修复方案
在加载模型之前，必须显式将这些类型加入安全白名单：
```python
import torch
import numpy as np
import numpy.dtypes

torch.serialization.add_safe_globals([
    np.core.multiarray.scalar, 
    np.dtype, 
    np.dtypes.Float64DType, 
    np.dtypes.Int64DType
])
```

## 2. RuntimeError: Missing key(s) in state_dict

### 问题描述
当从官方环境切换到自定义注册环境时，Sample Factory 可能会改变 `ObservationEncoder` 的内部命名空间。
- 报错：`Missing key(s): "encoder.basic_encoder.enc..."`
- 实际存在：`"encoder.encoders.obs.enc..."`

### 修复方案 (Monkey Patch)
拦截 `torch.load` 并在返回字典前进行字符串替换：
```python
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False # 强制关闭安全检查以允许修复
    checkpoint = original_torch_load(*args, **kwargs)
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        new_model_state = {}
        for k, v in checkpoint["model"].items():
            # 核心替换逻辑
            new_key = k.replace("encoder.encoders.obs.enc", "encoder.basic_encoder.enc")
            new_model_state[new_key] = v
        checkpoint["model"] = new_model_state
    return checkpoint

torch.load = patched_torch_load
```

## 3. 最佳实践
- 始终使用 [src/run_enjoy_safe.py](src/run_enjoy_safe.py) 进行评估。
- 如果在训练中遇到类似问题，可以将上述逻辑注入到训练脚本的入口处。
