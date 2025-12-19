import torch
import numpy as np
import numpy.dtypes  # 显式导入 dtypes 模块

# 将 numpy 的各种标量和类型类加入 PyTorch 加载白名单
try:
    safe_list = [
        np.core.multiarray.scalar,
        np.dtype,
        numpy.dtypes.Float64DType,  # 报错中明确提到的类
        numpy.dtypes.Int64DType,    # 预防性添加
    ]
    
    # 尝试添加 Sample Factory 的 AttrDict，这也是常见的拦截点
    try:
        from sample_factory.utils.utils import AttrDict
        safe_list.append(AttrDict)
    except ImportError:
        pass

    torch.serialization.add_safe_globals(safe_list)
    print(f"✅ 已更新 PyTorch 安全白名单: {[c.__name__ for c in safe_list if hasattr(c, '__name__')]}")
except Exception as e:
    print(f"⚠️ 注入白名单失败: {e}")

# 导入并运行原有的 enjoy 脚本
from sf_examples.vizdoom.enjoy_vizdoom import main

if __name__ == "__main__":
    # main() 会自动解析 sys.argv
    main()