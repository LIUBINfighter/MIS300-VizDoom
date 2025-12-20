import gymnasium as gym
import numpy as np
import cv2
import vizdoom as vzd
import types

class RewardShapingWrapper(gym.Wrapper):
    """
    奖励塑形包装器 - 狙击手版 (Sniper Edition)
    目标：消除乱开枪，鼓励精准点射。
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_vars = {}

    def _query_game_variable(self, var):
        """安全地查询底层游戏变量"""
        try:
            unwrapped = getattr(self.env, 'unwrapped', None)
            candidates = [unwrapped]
            if unwrapped is not None:
                if hasattr(unwrapped, 'game'): candidates.append(unwrapped.game)
                if hasattr(unwrapped, '_game'): candidates.append(unwrapped._game)
            
            for c in candidates:
                if c and hasattr(c, 'get_game_variable'):
                    try:
                        return c.get_game_variable(var)
                    except:
                        continue
        except:
            pass
        return 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 初始化变量记录
        self.prev_vars = {
            'KILLCOUNT': self._query_game_variable(vzd.GameVariable.KILLCOUNT),
            'HITCOUNT': self._query_game_variable(vzd.GameVariable.HITCOUNT),
            'HEALTH': self._query_game_variable(vzd.GameVariable.HEALTH),
            'AMMO2': self._query_game_variable(vzd.GameVariable.AMMO2),
        }
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # --- 1. 获取关键数据 ---
        curr_hits = self._query_game_variable(vzd.GameVariable.HITCOUNT)
        curr_kills = self._query_game_variable(vzd.GameVariable.KILLCOUNT)
        curr_health = self._query_game_variable(vzd.GameVariable.HEALTH)
        curr_ammo = self._query_game_variable(vzd.GameVariable.AMMO2)

        # 计算增量
        diff_hits = curr_hits - self.prev_vars.get('HITCOUNT', 0)
        diff_kills = curr_kills - self.prev_vars.get('KILLCOUNT', 0)
        diff_health = curr_health - self.prev_vars.get('HEALTH', 100)
        diff_ammo = self.prev_vars.get('AMMO2', 0) - curr_ammo # 消耗了多少子弹

        # --- 2. 奖励工程 (核心修改) ---

        # A. 基础生存奖励 (活着就是胜利)
        reward += 0.02 

        # B. 击杀奖励 (大奖)
        if diff_kills > 0:
            reward += 10.0 * diff_kills

        # C. 命中机制 (关键!)
        if diff_hits > 0:
            # 打中了！给予奖励
            reward += 2.0 * diff_hits
        
        # D. 开枪惩罚逻辑 (Sniper Logic)
        if diff_ammo > 0: # 如果这一帧消耗了子弹（说明开枪了）
            if diff_hits > 0:
                # 开枪且命中了：稍微抵消一点弹药消耗，鼓励有效射击
                reward += 0.5 
            else:
                # 开枪但没命中 (空枪)：重罚！
                # 之前的惩罚几乎为0，现在设为 -1.0
                # 这意味着开一枪空枪的代价，相当于丢了半条命，或者抵消了半次命中的奖励
                reward -= 1.0 
        
        # E. 掉血惩罚 (轻微，避免过于畏缩)
        if diff_health < 0:
            reward += 0.1 * diff_health # diff_health是负数

        # F. 弹药管理 (捡到子弹给奖励，防止没子弹干瞪眼)
        if curr_ammo > self.prev_vars.get('AMMO2', 0):
             reward += 0.5

        # --- 3. 更新状态 ---
        self.prev_vars = {
            'KILLCOUNT': curr_kills,
            'HITCOUNT': curr_hits,
            'HEALTH': curr_health,
            'AMMO2': curr_ammo,
        }

        # 注入 Info 供 Tensorboard 观察
        info['HIT_INC'] = diff_hits
        info['KILL_INC'] = diff_kills
        info['AMMO_USED'] = diff_ammo

        return obs, reward, terminated, truncated, info

class ImageCleaningWrapper(gym.ObservationWrapper):
    """
    修复视觉畸变：保持 4:3 比例，不强行拉伸
    """
    def __init__(self, env):
        super().__init__(env)
        # 目标分辨率：128x96 (4:3)
        self.w, self.h = 128, 96
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, self.h, self.w), dtype=np.float32
        )
        
    def observation(self, obs):
        # 统一格式 (C, H, W) -> (H, W, C)
        if obs.shape[0] == 3:
            obs = np.transpose(obs, (1, 2, 0))
            
        h, w, c = obs.shape
        
        # 1. 垂直裁剪 (切掉底部的状态栏)
        # 假设底部 15% 是状态栏，顶部切一点点
        top_crop = int(h * 0.05)
        bot_crop = int(h * 0.15)
        obs = obs[top_crop:h-bot_crop, :, :]
        
        # 2. 调整大小 (使用 INTER_AREA 抗锯齿)
        # 不再做中心裁剪，而是直接缩放到 128x96
        # 虽然这还是会有轻微拉伸（如果原图裁剪后不是4:3），但比之前压成正方形好得多
        obs = cv2.resize(obs, (self.w, self.h), interpolation=cv2.INTER_AREA)
        
        # 3. 归一化 & 转回 (C, H, W)
        obs = np.transpose(obs, (2, 0, 1))
        obs = obs.astype(np.float32) / 255.0
        return obs

class CompositeActionWrapper(gym.ActionWrapper):
    """
    动作组合器：让 Agent 可以一边转一边开火。
    [0:左, 1:右, 2:开火, 3:左+开火, 4:右+开火]
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(5) 

    def action(self, act):
        return act # 占位，实际逻辑在 step

    def step(self, action):
        # 映射表: [TURN_LEFT, TURN_RIGHT, ATTACK]
        # 注意：这里的 1 和 0 取决于你的 .cfg 文件中 available_buttons 的顺序
        # 通常顺序是: TURN_LEFT, TURN_RIGHT, ATTACK
        
        real_action = [0, 0, 0] # 默认不动
        
        if action == 0: real_action = [1, 0, 0]   # 左
        elif action == 1: real_action = [0, 1, 0] # 右
        elif action == 2: real_action = [0, 0, 1] # 开火
        elif action == 3: real_action = [1, 0, 1] # 左+开火 (扫射)
        elif action == 4: real_action = [0, 1, 1] # 右+开火 (扫射)
        
        # 极其关键：因为我们 hack 了底层的 step，需要判断底层接受什么
        # 我们的 CustomVizdoomEnv 已经修改为接受 list，所以直接传
        return self.env.step(real_action)