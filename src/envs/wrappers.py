import gymnasium as gym
import numpy as np
import cv2
import vizdoom as vzd
import types


class RewardShapingWrapper(gym.Wrapper):
    """
    自定义奖励包装器。
    针对 Defend Center 优化的版本。
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_vars = {}

    def _query_game_variable(self, var):
        """Try to read a game variable from the underlying VizDoom object via several common attributes.
        This is robust to multiple wrapper layers; it will traverse `.env`, `.unwrapped` and look for
        attributes like `game`, `_game`, or direct get_game_variable implementations.
        """
        try:
            candidates = []
            curr = self.env
            seen = set()
            # Traverse wrapper chain
            while curr is not None and id(curr) not in seen:
                seen.add(id(curr))
                # direct method on wrapper
                if hasattr(curr, 'get_game_variable') and isinstance(getattr(curr, 'get_game_variable'), types.MethodType):
                    candidates.append(curr)
                # attributes that may hold the underlying DoomGame
                if hasattr(curr, 'game'):
                    candidates.append(curr.game)
                if hasattr(curr, '_game'):
                    candidates.append(curr._game)
                # try moving deeper
                if hasattr(curr, 'unwrapped') and curr.unwrapped is not curr:
                    curr = curr.unwrapped
                elif hasattr(curr, 'env') and curr.env is not curr:
                    curr = curr.env
                else:
                    break

            for c in candidates:
                if c is None:
                    continue
                if hasattr(c, 'get_game_variable') and isinstance(getattr(c, 'get_game_variable'), types.MethodType):
                    try:
                        return c.get_game_variable(var)
                    except Exception:
                        continue
        except Exception:
            pass
        return 0

    def _get_available_game_variables(self):
        """Attempt to discover available game variables exposed by the underlying game object."""
        try:
            curr = self.env
            seen = set()
            while curr is not None and id(curr) not in seen:
                seen.add(id(curr))
                if hasattr(curr, 'get_available_game_variables') and isinstance(getattr(curr, 'get_available_game_variables'), types.MethodType):
                    try:
                        return curr.get_available_game_variables()
                    except Exception:
                        pass
                if hasattr(curr, 'game') and hasattr(curr.game, 'get_available_game_variables'):
                    try:
                        return curr.game.get_available_game_variables()
                    except Exception:
                        pass
                if hasattr(curr, 'unwrapped') and curr.unwrapped is not curr:
                    curr = curr.unwrapped
                elif hasattr(curr, 'env') and curr.env is not curr:
                    curr = curr.env
                else:
                    break
        except Exception:
            pass
        return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 记录初始状态，默认 AMMO2 设为 0（更稳健）
        # Use _query_game_variable as a fallback when info doesn't contain values
        self.prev_vars = {
            'KILLCOUNT': info.get('KILLCOUNT', self._query_game_variable(vzd.GameVariable.FRAGCOUNT)),
            'HITCOUNT': info.get('HITCOUNT', self._query_game_variable(vzd.GameVariable.HITCOUNT)),
            'HEALTH': info.get('HEALTH', self._query_game_variable(vzd.GameVariable.HEALTH)),
            'AMMO2': info.get('AMMO2', self._query_game_variable(vzd.GameVariable.AMMO2)),
        }
        # 如果 HITCOUNT 不存在，尝试查询底层 game 的可用变量并给出更精确的提示
        if 'HITCOUNT' not in info:
            available = self._get_available_game_variables()
            if available:
                print(f"[Warning] HITCOUNT not in env.info. Underlying game reports available variables: {available}. If HITCOUNT absent, add it to scenario's available_game_variables.")
            else:
                print("[Warning] HITCOUNT not present in env.info and could not detect underlying game's available variables. Ensure 'available_game_variables = { KILLCOUNT HITCOUNT AMMO2 HEALTH FRAGCOUNT }' is present in the scenario .cfg and restart.")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- 基础生存奖励 (轻微正向，鼓励存活) ---
        reward += 0.01

        # --- 命中奖励 (比之前更显著，鼓励瞄准) ---
        # 如果 info 中没有 HITCOUNT，则尝试从底层 DoomGame 查询
        current_hits = info.get('HITCOUNT', None)
        if current_hits is None:
            current_hits = self._query_game_variable(vzd.GameVariable.HITCOUNT)
        diff_hits = current_hits - self.prev_vars.get('HITCOUNT', 0)
        if diff_hits > 0:
            reward += 5.0 * diff_hits

        # --- 击杀奖励 ---
        current_kills = info.get('KILLCOUNT', None)
        if current_kills is None:
            # FRAGCOUNT is equivalent to KILLCOUNT in some scenarios
            current_kills = self._query_game_variable(vzd.GameVariable.FRAGCOUNT)
        diff_kills = current_kills - self.prev_vars.get('KILLCOUNT', 0)
        if diff_kills > 0:
            reward += 15.0 * diff_kills

        # --- 弹药减少（射击）惩罚（非常轻微） ---
        prev_ammo = self.prev_vars.get('AMMO2', 0)
        current_ammo = info.get('AMMO2', prev_ammo)
        diff_ammo = prev_ammo - current_ammo
        if diff_ammo > 0:
            # 如果开火了 (diff_ammo > 0)
            if diff_hits > 0:
                # 且击中了 -> 奖励抵消消耗，甚至额外奖励
                reward += 0.5 
            else:
                # 开火了但没击中 -> 空枪惩罚！
                # 这会迫使 Agent 只有在有把握（瞄准了）的时候才开枪
                reward -= 0.1 

        # --- 掉血惩罚（降低权重，避免过于胆小） ---
        current_health = info.get('HEALTH', 100)
        diff_health = current_health - self.prev_vars.get('HEALTH', 100)
        if diff_health < 0:
            reward += 0.05 * diff_health  # 负值

        # 更新记录
        self.prev_vars.update({
            'KILLCOUNT': current_kills,
            'HITCOUNT': current_hits,
            'HEALTH': current_health,
            'AMMO2': current_ammo,
        })

        # Augment info with incremental metrics for logging (SF / TB / CSV friendly)
        try:
            info['HIT_INC'] = int(diff_hits)
            info['KILL_INC'] = int(diff_kills)
            info['AMMO_USED'] = int(max(0, prev_ammo - current_ammo))
        except Exception:
            # Ensure we never break env contract
            pass
        
        return obs, reward, terminated, truncated, info

class ImageCleaningWrapper(gym.ObservationWrapper):
    """
    图像预处理包装器。
    实现：垂直裁剪 (移除HUD)、保持长宽比缩放、归一化。
    
    【重要修复】不再强制拉伸为正方形 (112x112)，而是使用 4:3 比例 (128x96)。
    原因：强制拉伸会破坏几何特征，导致 Agent 难以学习精确瞄准。
    使用 cv2.INTER_AREA 在缩小时保留更多细节。
    """
    def __init__(self, env):
        super().__init__(env)
        # 使用 4:3 的分辨率，128x96 是一个平衡了性能和细节的选择
        self.w, self.h = 128, 96
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, self.h, self.w), dtype=np.float32
        )
        
    def observation(self, obs):
        # VizDoom 出来的 obs 可能是 (H, W, 3) 或者是 (3, H, W)
        # 统一处理为 HWC 进行裁剪和缩放
        if obs.shape[0] == 3:
            obs = np.transpose(obs, (1, 2, 0))
            
        h, w, c = obs.shape
        
        # 1. 垂直裁剪 (Vertical Crop)
        # 顶部少裁一点 (5%)，底部移除 HUD (15%)
        top_cut = int(h * 0.05)
        bot_cut = int(h * 0.15)
        obs = obs[top_cut:h-bot_cut, :, :]
        
        # 2. 缩放 (保持 4:3 比例)
        # 使用 cv2.INTER_AREA 可以在缩小时保留更多细节（抗锯齿）
        obs = cv2.resize(obs, (self.w, self.h), interpolation=cv2.INTER_AREA)
        
        # 3. 调整维度顺序 HWC -> CHW (PyTorch 需要)
        obs = np.transpose(obs, (2, 0, 1))
        
        # 4. 归一化
        obs = obs.astype(np.float32) / 255.0
        
        return obs
class CompositeActionWrapper(gym.ActionWrapper):
    """
    将原始的离散动作空间扩展，包含组合动作。
    针对 Defend the Center 优化：允许同时转头和开火。
    """
    def __init__(self, env):
        super().__init__(env)
        # 假设原始环境有 3 个动作: [左转, 右转, 开火]
        self.action_space = gym.spaces.Discrete(5)

        # 试图从底层 env 中获取动作映射（如 VizDoomEnv.actions），以便我们能把按钮向量映射
        # 回底层的离散动作索引（避免直接传 list 导致类型错误）。
        self._action_mapping = None
        try:
            base = env
            while hasattr(base, 'env'):
                base = base.env
            # 一些实现使用 'actions' 或 '_actions' 来保存离散动作列表
            actions_list = None
            if hasattr(base, 'actions'):
                actions_list = getattr(base, 'actions')
            elif hasattr(base, '_actions'):
                actions_list = getattr(base, '_actions')

            if actions_list is not None:
                # 以 tuple 作为键建立映射
                self._action_mapping = {tuple(a): i for i, a in enumerate(actions_list)}
                print(f"[Debug] CompositeActionWrapper built mapping of {len(self._action_mapping)} actions")
            else:
                print("[Warning] Could not find underlying discrete actions list; CompositeActionWrapper will attempt fallbacks.")
        except Exception as e:
            print(f"[Warning] Error while building action mapping: {e}")

    def action(self, act):
        # 这里的映射逻辑需要与 step 中的 real_action 对应
        return act 

    def step(self, action):
        # 将离散 ID 转换为 VizDoom 的按钮列表 [TURN_LEFT, TURN_RIGHT, ATTACK]
        real_action = [0, 0, 0]

        if action == 0:
            real_action = [1, 0, 0]  # 左转
        elif action == 1:
            real_action = [0, 1, 0]  # 右转
        elif action == 2:
            real_action = [0, 0, 1]  # 开火
        elif action == 3:
            real_action = [1, 0, 1]  # 左转 + 开火
        elif action == 4:
            real_action = [0, 1, 1]  # 右转 + 开火

        # 如果我们有底层的动作映射，则优先使用映射的离散索引（整数），以匹配底层接口期望
        if self._action_mapping is not None:
            key = tuple(int(x) for x in real_action)
            mapped = self._action_mapping.get(key, None)
            if mapped is not None:
                return self.env.step(mapped)
            else:
                # 如果映射中没有精确项，则尝试最接近的匹配（Hamming 距离最小）
                best_k = None
                best_d = None
                for k in self._action_mapping.keys():
                    d = sum(a != b for a, b in zip(k, key))
                    if best_d is None or d < best_d:
                        best_d = d
                        best_k = k
                if best_k is not None:
                    mapped = self._action_mapping[best_k]
                    print(f"[Debug] CompositeActionWrapper: no exact mapping for {key}, using nearest action {best_k} (idx {mapped})")
                    return self.env.step(mapped)
                else:
                    print("[Warning] No mapping available; falling back to sending raw button list (may fail).")

        # 最后备选：尝试发送 numpy array（许多底层会接受 numpy array），或者 tuple
        try:
            import numpy as _np
            return self.env.step(_np.array(real_action, dtype=_np.int8))
        except Exception:
            try:
                return self.env.step(tuple(real_action))
            except Exception as e:
                # 最后还是抛出异常，保留原始行为
                raise e
