import sys
from sample_factory.train import run_rl
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults

# 关键：导入 src.envs 以触发环境注册
import src.envs 
from src.models import register_models

def main():
    """
    自定义训练入口。
    使用方式与 sf_examples.vizdoom.train_vizdoom 类似，
    但支持我们注册在 src.envs 中的 custom_doom_* 环境。
    """
    # 注册自定义模型
    register_models()

    parser, cfg = parse_sf_args(evaluation=False)
    
    # 添加 ViZDoom 特有参数
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    
    # 强制修改默认参数
    parser.set_defaults(
        model="custom_vizdoom_model",
        # 【新增】叠加最近4帧，让 Agent 能感知"速度"和运动方向
        frame_stack=4, 
        # 【明确设置】env_frameskip=1（更细粒度控制，以利于精准瞄准）
        # frameskip=1 提供最高操作频率，适合需要精确操作的任务
        env_frameskip=1,
        # 【调整】稍微提高学习率，加快初期收敛
        learning_rate=0.0002,
        # 【关键修改】大幅提高熵系数，增强探索能力，防止陷入局部最优
        entropy_coeff=0.01 
    )
    
    cfg = parse_full_cfg(parser)
    
    status = run_rl(cfg)
    return status

if __name__ == "__main__":
    sys.exit(main())
