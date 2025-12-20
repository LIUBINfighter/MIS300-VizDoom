import gymnasium as gym
from src.envs.vizdoom_env import create_vizdoom_env

def test_env():
    env_name = "custom_doom_defend_the_center"
    env = create_vizdoom_env(env_name)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Inspect underlying env
    curr = env
    while hasattr(curr, 'env'):
        print(f"Wrapper: {type(curr)}")
        # 如果底层 wrapper 暴露了 'actions' 列表，打印前 5 个以免输出过长
        if hasattr(curr, 'actions'):
            print(f"Found actions (sample): {curr.actions[:5]}")
        curr = curr.env
    print(f"Base env: {type(curr)}")
    if hasattr(curr, 'actions'):
        print(f"Base env actions (sample): {curr.actions[:5]}")
    
    obs, info = env.reset()
    # ...

if __name__ == "__main__":
    test_env()
