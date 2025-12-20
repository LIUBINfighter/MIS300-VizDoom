import cv2
import numpy as np
import os
import torch
from src.envs.vizdoom_env import create_vizdoom_env

def debug_agent_vision():
    # 1. 创建带有 Wrapper 的环境
    # 注意：这里需要确保环境能正常创建，可能需要传入一些默认 cfg
    env = create_vizdoom_env("custom_doom_defend_the_center")
    
    obs, info = env.reset()
    
    # 2. 保存前 20 帧处理后的画面
    output_dir = "dist/debug_vision"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📸 Saving agent's actual observation (84x84) to {output_dir}...")
    
    for i in range(20):
        # 随机动作让画面动起来
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        
        # obs 是 [Channel, Height, Width] -> [1, 84, 84] (如果是灰度)
        # 或者是 [4, 84, 84] (如果开了 Stack)
        
        # 我们只看最新的一帧 (obs[0] 或 obs[-1])
        if len(obs.shape) == 3:
            # 如果是 (C, H, W)
            img_data = obs[0] if obs.shape[0] == 1 else obs[-1] 
        else:
            img_data = obs
            
        # 反归一化：从 [0, 1]变回 [0, 255]
        img_visual = (img_data * 255.0).astype(np.uint8)
        
        # 保存图片
        cv2.imwrite(f"{output_dir}/frame_{i:03d}.png", img_visual)
        
    print(f"✅ Done. Check the '{output_dir}' folder.")
    print("   Does the enemy look distinct from the wall?")
    env.close()

if __name__ == "__main__":
    debug_agent_vision()
