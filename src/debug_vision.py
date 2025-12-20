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
    
    print(f"📸 Saving agent's actual observation to {output_dir}...")
    
    for i in range(20):
        # 随机动作让画面动起来
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        
        # obs 是 [Channel, Height, Width]
        # 我们只看最新的一帧 (obs[0] 或 obs[-1])
        if len(obs.shape) == 3:
            # 如果是 (C, H, W)
            img_data = obs # 保持 (C, H, W)
        else:
            # 异常情况处理
            img_data = obs
            
        # 转换为 HWC 格式用于 OpenCV
        if img_data.shape[0] == 3:
            img_visual = np.transpose(img_data, (1, 2, 0))
        else:
            img_visual = img_data
            
        # 反归一化：从 [0, 1] 变回 [0, 255]
        img_visual = (img_visual * 255.0).astype(np.uint8)
        
        # 确保是 BGR 格式 (如果是 RGB)
        img_visual = cv2.cvtColor(img_visual, cv2.COLOR_RGB2BGR)
        
        # --- 关键：画一个中心十字线验证准心 ---
        h, w = img_visual.shape[:2]
        cx, cy = w // 2, h // 2
        # 绿色十字 (B, G, R)
        cv2.line(img_visual, (cx - 5, cy), (cx + 5, cy), (0, 255, 0), 1)
        cv2.line(img_visual, (cx, cy - 5), (cx, cy + 5), (0, 255, 0), 1)
        
        # 保存图片
        cv2.imwrite(f"{output_dir}/frame_{i:03d}.png", img_visual)
        
    print(f"✅ Done. Check the '{output_dir}' folder.")
    print("   Does the enemy look distinct from the wall?")
    env.close()

if __name__ == "__main__":
    debug_agent_vision()
