import gymnasium as gym
import numpy as np

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
    例如：灰度化、边缘检测、裁剪等。
    """
    def __init__(self, env):
        super().__init__(env)
        # 如果修改了观察空间的大小或通道，需要更新 self.observation_space
        
    def observation(self, obs):
        # 示例：简单的归一化或处理
        return obs
