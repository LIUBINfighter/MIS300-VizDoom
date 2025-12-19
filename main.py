import sys

import gymnasium as gym
import vizdoom as vzd
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import run_rl


def check_env():
    print("--- 🏁 开始环境验收 ---")

    # 1. 验证 ViZDoom 基础功能
    try:
        game = vzd.DoomGame()
        # 这里使用一个内置的默认配置
        game.load_config(vzd.scenarios_path + "/basic.cfg")
        game.set_window_visible(False)  # Docker 内部必须为 False
        game.init()
        print(
            f"✅ ViZDoom 初始化成功! 场景分辨率: {game.get_screen_width()}x{game.get_screen_height()}"
        )
        game.close()
    except Exception as e:
        print(f"❌ ViZDoom 初始化失败: {e}")
        return False

    # 2. 验证依赖库导入
    try:
        import torch

        print(f"✅ PyTorch 版本: {torch.__version__}")
        print(f"✅ CUDA 是否可用: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ 核心依赖库缺失")
        return False

    print("--- 🎉 所有基准检查已通过! ---")
    return True


if __name__ == "__main__":
    if check_env():
        sys.exit(0)
    else:
        sys.exit(1)
