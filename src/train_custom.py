import sys
from sample_factory.train import run_rl
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults

# 关键：导入 src.envs 以触发环境注册
import src.envs 

def main():
    """
    自定义训练入口。
    使用方式与 sf_examples.vizdoom.train_vizdoom 类似，
    但支持我们注册在 src.envs 中的 custom_doom_* 环境。
    """
    parser, cfg = parse_sf_args(evaluation=False)
    
    # 添加 ViZDoom 特有参数
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    
    cfg = parse_full_cfg(parser)
    
    status = run_rl(cfg)
    return status

if __name__ == "__main__":
    sys.exit(main())
