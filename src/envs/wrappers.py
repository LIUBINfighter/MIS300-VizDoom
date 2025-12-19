import gymnasium as gym
import numpy as np
import cv2

class RewardShapingWrapper(gym.Wrapper):
    """
    自定义奖励包装器。
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_vars = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 记录初始状态
        self.prev_vars = {
            'KILLCOUNT': info.get('KILLCOUNT', 0),
            'HEALTH': info.get('HEALTH', 100),
            'AMMO2': info.get('AMMO2', 0),
            'POSITION_X': info.get('POSITION_X', 0),
        }
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. 击杀奖励 (针对所有场景)
        current_kills = info.get('KILLCOUNT', 0)
        diff_kills = current_kills - self.prev_vars.get('KILLCOUNT', 0)
        if diff_kills > 0:
            reward += 20.0 * diff_kills # 死亡走廊杀敌更难，给 20 分

        # 2. 前进奖励 (专门针对死亡走廊)
        # 假设走廊是沿 X 轴延伸的
        current_x = info.get('POSITION_X', 0)
        diff_x = current_x - self.prev_vars.get('POSITION_X', 0)
        if diff_x > 0:
            reward += diff_x * 0.01 # 鼓励向前走

        # 3. 掉血惩罚
        current_health = info.get('HEALTH', 100)
        diff_health = current_health - self.prev_vars.get('HEALTH', 100)
        if diff_health < 0:
            reward += 0.5 * diff_health # 死亡走廊子弹多，惩罚重一点

        # 4. 弹药惩罚
        current_ammo = info.get('AMMO2', 0)
        if current_ammo < self.prev_vars.get('AMMO2', 0):
            reward -= 0.2

        self.prev_vars.update({
            'KILLCOUNT': current_kills,
            'HEALTH': current_health,
            'AMMO2': current_ammo,
            'POSITION_X': current_x,
        })
        
        return obs, reward, terminated, truncated, info

class ImageCleaningWrapper(gym.ObservationWrapper):
    """
    图像预处理包装器。
    实现：裁剪状态栏、灰度化、缩放至 84x84、归一化。
    """
    def __init__(self, env):
        super().__init__(env)
        # 更新观察空间为 (1, 84, 84)，归一化后的 float32
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1, 84, 84), dtype=np.float32
        )
        
    def observation(self, obs):
        # 1. 灰度化 (如果是 RGB)
        if len(obs.shape) == 3 and obs.shape[0] == 3: # CHW
            obs = np.transpose(obs, (1, 2, 0)) # HWC
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif len(obs.shape) == 3 and obs.shape[2] == 3: # HWC
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            
        # 2. 裁剪状态栏 (VizDoom 默认底部约 15-20% 是状态栏)
        # 假设原始高度是 240，状态栏大约在 200 之后
        h, w = obs.shape
        obs = obs[:int(h * 0.85), :]
        
        # 3. 缩放至 84x84
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        
        # 4. 归一化并增加通道维度 (1, 84, 84)
        obs = obs.astype(np.float32) / 255.0
        obs = np.expand_dims(obs, axis=0)
        
        return obs
