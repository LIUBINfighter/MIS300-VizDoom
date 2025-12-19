import gymnasium as gym
import numpy as np

class RewardShapingWrapper(gym.Wrapper):
    """
    自定义奖励包装器。
    通过 info 中的游戏变量来计算额外的奖励或惩罚。
    Sample Factory 会将 ViZDoom 的游戏变量直接放入 info 字典中。
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_vars = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 记录初始状态
        self.prev_vars = {
            'FRAGCOUNT': info.get('FRAGCOUNT', 0),
            'HEALTH': info.get('HEALTH', 100),
            'AMMO2': info.get('AMMO2', 0),
        }
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. 杀敌奖励 (Frag reward)
        # 注意：有些场景可能叫 FRAGCOUNT，有些可能需要自定义
        current_frags = info.get('FRAGCOUNT', info.get('USER1', 0))
        diff_frags = current_frags - self.prev_vars.get('FRAGCOUNT', 0)
        if diff_frags > 0:
            reward += 10.0 * diff_frags
            # print(f"Kill detected! Reward +10. Total reward: {reward}")
            
        # 2. 掉血惩罚 (Health penalty)
        current_health = info.get('HEALTH', 100)
        diff_health = current_health - self.prev_vars.get('HEALTH', 100)
        if diff_health < 0:
            reward += 0.1 * diff_health # diff_health 是负数
            
        # 3. 弹药消耗惩罚 (可选)
        # current_ammo = info.get('AMMO2', 0)
        # diff_ammo = current_ammo - self.prev_vars.get('AMMO2', 0)
        # if diff_ammo < 0:
        #     reward -= 0.01 # 每次开火轻微惩罚，鼓励精准度
            
        # 更新记录
        self.prev_vars['FRAGCOUNT'] = current_frags
        self.prev_vars['HEALTH'] = current_health
        # self.prev_vars['AMMO2'] = current_ammo
        
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
