import gymnasium as gym
import numpy as np
import cv2

class RewardShapingWrapper(gym.Wrapper):
    """
    自定义奖励包装器。
    针对 Defend Center 优化的版本。
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
            'AMMO2': info.get('AMMO2', 50), # 确保默认值合理
        }
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- 修复 1: 大幅增加杀敌奖励 ---
        # 原始奖励可能太小，不足以抵消掉血的惩罚
        current_kills = info.get('KILLCOUNT', 0)
        diff_kills = current_kills - self.prev_vars.get('KILLCOUNT', 0)
        
        if diff_kills > 0:
            # 杀敌是唯一目标，给予巨额奖励，让 Agent "上瘾"
            reward += 1.0  # 假设 VizDoom 原始配置已经给了分，这里额外加分
            # 注意：如果原始配置没给分，这里建议给 +100.0

        # --- 修复 2: 移除“位置移动”奖励 ---
        # Defend the Center 是圆周运动，X轴位移奖励是噪音，必须删除。

        # --- 修复 3: 移除“开枪惩罚” ---
        # 只有在高级阶段才需要节省弹药。初期必须鼓励开枪。
        
        # --- 4. 掉血惩罚 (保留但减小权重) ---
        # 让 Agent 稍微在意一下血量，但不要让它怕到不敢动
        current_health = info.get('HEALTH', 100)
        diff_health = current_health - self.prev_vars.get('HEALTH', 100)
        if diff_health < 0:
            reward += 0.05 * diff_health  # 降低惩罚权重，避免 Agent 绝望

        self.prev_vars.update({
            'KILLCOUNT': current_kills,
            'HEALTH': current_health,
            'AMMO2': info.get('AMMO2', 0),
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
