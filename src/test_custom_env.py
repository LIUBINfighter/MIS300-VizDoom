import gymnasium as gym
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
# 导入 ViZDoom 特有的参数处理函数
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
import src.envs # 触发注册

def test_env_creation():
    env_name = "custom_doom_defend_the_center"
    print(f"Testing creation of {env_name}...")
    
    # 1. 获取基础参数解析器
    parser, cfg = parse_sf_args(evaluation=True)
    
    # 2. 添加 ViZDoom 特有参数 (修复 AttributeError 的关键)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    
    # 3. 解析最终配置
    cfg.env = env_name
    cfg = parse_full_cfg(parser)
    
    # 4. 创建环境
    from src.envs.vizdoom_env import create_vizdoom_env
    env = create_vizdoom_env(env_name, cfg, env_config=None)
    
    print("Environment created successfully!")
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    
    # 跑一步
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"Step reward: {reward}")
    print(f"Step info keys: {info.keys()}")
    
    env.close()
    print("Test passed!")

if __name__ == "__main__":
    test_env_creation()
