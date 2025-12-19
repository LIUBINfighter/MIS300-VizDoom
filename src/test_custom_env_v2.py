import gymnasium as gym
from src.envs.vizdoom_env import create_vizdoom_env
import numpy as np

def test_custom_env():
    print("Testing custom VizDoom environment...")
    
    # 使用我们修改后的场景名
    env_name = "custom_doom_health_gathering"
    
    try:
        env = create_vizdoom_env(env_name, render_mode=None)
        print(f"Environment {env_name} created successfully.")
        
        # 验证观察空间
        print(f"Observation space: {env.observation_space}")
        assert env.observation_space.shape == (1, 84, 84), f"Expected (1, 84, 84), got {env.observation_space.shape}"
        
        obs, info = env.reset()
        print(f"Reset observation shape: {obs.shape}")
        assert obs.shape == (1, 84, 84), f"Expected (1, 84, 84), got {obs.shape}"
        
        # 执行一步
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step observation shape: {obs.shape}, reward: {reward}")
        
        print("✓ Phase 1 Verification Successful!")
        env.close()
        
    except Exception as e:
        print(f"Error during environment testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_custom_env()
