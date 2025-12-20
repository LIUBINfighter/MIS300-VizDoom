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
        # 【新增】叠加最近4帧，让 Agent 能感知“速度”
        frame_stack=4, 
        # 【新增】调整学习率，默认的可能太高或太低
        learning_rate=0.0001,
        # 【建议】降低熵系数，现在的 0.04 太高了，导致 Agent 行为过于随机
        entropy_coeff=0.001 
    )
    
    cfg = parse_full_cfg(parser)
    
    status = run_rl(cfg)
    return status

if __name__ == "__main__":
    sys.exit(main())
