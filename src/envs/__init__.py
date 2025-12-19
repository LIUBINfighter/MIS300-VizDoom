from sample_factory.envs.env_utils import register_env
from src.envs.vizdoom_env import create_vizdoom_env

def register_custom_doom_envs():
    """
    将自定义环境注册到 Sample Factory。
    """
    custom_envs = [
        "custom_doom_basic",
        "custom_doom_defend_the_center",
        "custom_doom_deadly_corridor",
        "custom_doom_health_gathering",
    ]
    
    for env_name in custom_envs:
        register_env(env_name, create_vizdoom_env)
        print(f"Registered custom env: {env_name}")

# 自动执行注册
register_custom_doom_envs()
