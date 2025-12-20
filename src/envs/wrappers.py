import gymnasium as gym
import numpy as np
import cv2
import vizdoom as vzd
import types

class RewardShapingWrapper(gym.Wrapper):
    """
    奖励塑形包装器 - 狙击精英版 (Sniper Elite Edition)
    特性：
    1. 极度厌恶空枪 (Anti-Spray)
    2. 动作平滑约束 (Anti-Jitter)
    3. 鼓励爆头/击杀
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_vars = {}
        self.last_action_idx = 0 # 记录上一步动作，用于计算平滑度

    def _query_game_variable(self, var):
        """安全查询底层变量"""
        try:
            unwrapped = getattr(self.env, 'unwrapped', None)
            candidates = [unwrapped]
            if unwrapped is not None:
                if hasattr(unwrapped, 'game'): candidates.append(unwrapped.game)
                if hasattr(unwrapped, '_game'): candidates.append(unwrapped._game)
            for c in candidates:
                if c and hasattr(c, 'get_game_variable'):
                    try:
                        val = c.get_game_variable(var)
                        # 🚨 防御性检查：确保不会返回 None（防止 Worker 因 NoneType 崩溃）
                        if val is None:
                            return 0.0
                        return val
                    except:
                        continue
        except:
            pass
        return 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_vars = {
            'KILLCOUNT': self._query_game_variable(vzd.GameVariable.KILLCOUNT),
            'HITCOUNT': self._query_game_variable(vzd.GameVariable.HITCOUNT),
            'HEALTH': self._query_game_variable(vzd.GameVariable.HEALTH),
            'AMMO2': self._query_game_variable(vzd.GameVariable.AMMO2),
        }
        self.last_action_idx = 0
        return obs, info

    def step(self, action):
        # action 是一个 int (0~4)
        obs, reward, terminated, truncated, info = self.env.step(action)

        # --- 1. 获取状态 ---
        curr_hits = self._query_game_variable(vzd.GameVariable.HITCOUNT)
        curr_kills = self._query_game_variable(vzd.GameVariable.KILLCOUNT)
        curr_health = self._query_game_variable(vzd.GameVariable.HEALTH)
        curr_ammo = self._query_game_variable(vzd.GameVariable.AMMO2)

        diff_hits = curr_hits - self.prev_vars.get('HITCOUNT', 0)
        diff_kills = curr_kills - self.prev_vars.get('KILLCOUNT', 0)
        diff_ammo = self.prev_vars.get('AMMO2', 0) - curr_ammo

        # --- 2. 核心奖励逻辑 ---

        # A. 基础生存 (活着就好，但不要太高，否则它会选择苟着)
        reward += 0.01 

        # B. 击杀与命中 (重赏)
        if diff_kills > 0:
            reward += 10.0 * diff_kills  # 杀敌是大目标
        
        if diff_hits > 0:
            reward += 5.0 * diff_hits    # 命中是过程奖励（加重）

        # C. 严厉的空枪惩罚 (Sniper Discipline)
        # 只要消耗了子弹 (diff_ammo > 0) 且 没有命中 (diff_hits == 0)
        # 就视为浪费。
        if diff_ammo > 0:
            if diff_hits > 0:
                reward += 0.5  # 有效射击，抵消消耗
            else:
                # 空枪！适当惩罚，不要罚死
                # 这个值保持惩戒作用，但不会阻碍试错
                reward -= 0.5  

        # D. 动作稳定性惩罚 (Anti-Jitter)
        # 假设动作定义: 0:左, 1:右, 2:攻 ...
        # 如果上一步是左(0)，这一步是右(1)，说明在抖动。
        # 如果上一步是右(1)，这一步是左(0)，说明在抖动。
        if (self.last_action_idx == 0 and action == 1) or \
           (self.last_action_idx == 1 and action == 0):
            reward -= 0.5 # 抖动惩罚
        
        # 记录当前动作为下一步做参考
        self.last_action_idx = action

        # E. 掉血惩罚 (稍微轻一点，让它敢于对枪)
        diff_health = curr_health - self.prev_vars.get('HEALTH', 100)
        if diff_health < 0:
            reward += 0.05 * diff_health # 掉血微惩罚

        # --- 3. 更新状态 ---
        self.prev_vars = {
            'KILLCOUNT': curr_kills,
            'HITCOUNT': curr_hits,
            'HEALTH': curr_health,
            'AMMO2': curr_ammo,
        }
        
        # Log info
        info['HIT_INC'] = diff_hits
        info['AMMO_USED'] = diff_ammo

        return obs, reward, terminated, truncated, info

class ImageCleaningWrapper(gym.ObservationWrapper):
    """
    修复视觉畸变：保持 4:3 比例，不强行拉伸
    """
    def __init__(self, env):
        super().__init__(env)
        # 使用 128x96 (4:3)，这样 Agent 看到的画面几何比例是正常的
        # 有助于它判断角速度
        self.w, self.h = 128, 96
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, self.h, self.w), dtype=np.float32
        )
        
    def observation(self, obs):
        if obs.shape[0] == 3:
            obs = np.transpose(obs, (1, 2, 0))
        h, w, c = obs.shape
        
        # 垂直裁剪去除 HUD
        top_crop = int(h * 0.05)
        bot_crop = int(h * 0.15)
        obs = obs[top_crop:h-bot_crop, :, :]
        
        # 缩放 (保持比例)
        obs = cv2.resize(obs, (self.w, self.h), interpolation=cv2.INTER_AREA)
        
        obs = np.transpose(obs, (2, 0, 1))
        obs = obs.astype(np.float32) / 255.0
        return obs

class CompositeActionWrapper(gym.ActionWrapper):
    """
    [0:左, 1:右, 2:开火, 3:左+开火, 4:右+开火]
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(5) 

    def action(self, act):
        return act 

    def step(self, action):
        real_action = [0, 0, 0] 
        if action == 0: real_action = [1, 0, 0]   # 左
        elif action == 1: real_action = [0, 1, 0] # 右
        elif action == 2: real_action = [0, 0, 1] # 开火
        elif action == 3: real_action = [1, 0, 1] # 左+开火
        elif action == 4: real_action = [0, 1, 1] # 右+开火
        return self.env.step(real_action)