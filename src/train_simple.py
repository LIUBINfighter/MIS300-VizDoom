#!/usr/bin/env python
"""
简化版 ViZDoom 训练脚本
使用 Sample Factory 进行强化学习训练
目标：不报错 + 能跑通 + 有反馈
"""
import sys
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec


def register_custom_doom_env():
    """注册 ViZDoom 训练环境"""
    env_name = "doom_defend_simple"

    # 使用内置场景，简化配置
    env_spec = {
        "env_name": env_name,
        "scenario": "defend_the_center",  # 使用内置场景
    }

    def make_env(cfg):
        return make_doom_env_from_spec(env_spec, cfg)

    register_env(env_name, make_env)
    return env_name


def main():
    """主训练流程"""
    print("=== 开始 ViZDoom 训练 ===\n")

    # 先注册环境，并设置解析器默认值，避免 --env 必填报错
    env_name = register_custom_doom_env()

    # 解析参数
    parser, partial_cfg = parse_sf_args()
    add_doom_env_args(parser)
    # 关键：为解析器提供默认 env / experiment，避免必填参数阻塞
    parser.set_defaults(env=env_name)
    parser.set_defaults(experiment="defend_simple_test")
    doom_override_defaults(partial_cfg)

    # 配置训练参数（简化版）
    cfg = parse_full_cfg(parser)

    # 最小化资源配置，确保能跑通
    cfg.num_workers = 2              # 减少worker数
    cfg.num_envs_per_worker = 2      # 每个worker只运行2个环境
    cfg.train_for_env_steps = 100000 # 减少总步数，快速验证
    cfg.batch_size = 512             # 减小批次大小

    # 基础超参数
    cfg.learning_rate = 0.0001
    cfg.gamma = 0.99

    # 保存和日志
    cfg.save_every_sec = 120         # 每2分钟保存
    cfg.with_wandb = False           # 不使用 wandb

    print(f"训练配置:")
    print(f"  环境: {cfg.env}")
    print(f"  实验名: {cfg.experiment}")
    print(f"  总训练步数: {cfg.train_for_env_steps:,}")
    print(f"  并行环境数: {cfg.num_workers * cfg.num_envs_per_worker}")
    print()

    # 启动训练
    try:
        status = run_rl(cfg)
        print(f"\n=== 训练完成，退出码: {status} ===")
        return status
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
