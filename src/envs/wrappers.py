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
        }
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. 击杀奖励 (Kill reward)
        current_kills = info.get('KILLCOUNT', 0)
        diff_kills = current_kills - self.prev_vars.get('KILLCOUNT', 0)
        if diff_kills > 0:
            reward += 15.0 * diff_kills # 增加击杀权重
            
        # 2. 掉血惩罚 (Health penalty)
        current_health = info.get('HEALTH', 100)
        diff_health = current_health - self.prev_vars.get('HEALTH', 100)
        if diff_health < 0:
            reward += 0.2 * diff_health # 增加受伤惩罚
            
        # 3. 弹药消耗惩罚 (鼓励精准度)
        current_ammo = info.get('AMMO2', 0)
        diff_ammo = current_ammo - self.prev_vars.get('AMMO2', 0)
        if diff_ammo < 0:
            reward -= 0.1 # 每次开火扣除微量分数
            
        # 更新记录
        self.prev_vars['KILLCOUNT'] = current_kills
        self.prev_vars['HEALTH'] = current_health
        self.prev_vars['AMMO2'] = current_ammo
        
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
