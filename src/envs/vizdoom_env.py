import os
import vizdoom as vzd
import gymnasium as gym
import numpy as np
from sample_factory.envs.env_utils import register_env
from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec, DoomSpec, DOOM_ENVS
from sf_examples.vizdoom.doom.doom_gym import VizdoomEnv
from src.envs.wrappers import RewardShapingWrapper, ImageCleaningWrapper, CompositeActionWrapper

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__



def get_spec_by_scenario(scenario_name):
    for spec in DOOM_ENVS:
        if spec.env_spec_file == scenario_name:
            return spec
    return None

def create_vizdoom_env(env_name, cfg=None, env_config=None, render_mode=None, **kwargs):
    if cfg is None:
        # 在创建任何 VizDoom 对象之前设置环境变量以抑制 PipeWire/OpenAL 的噪声
        # 这会让音频驱动使用空驱动，避免 pipewire 的配置加载错误
        os.environ.setdefault('ALSOFT_DRIVERS', 'null')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

        cfg = AttrDict(
            record_to=None, 
            env_frameskip=2, 
            wide_aspect_ratio=False,
            res_w=160,
            res_h=120
        )

    scenarios = {
        "custom_doom_basic": "basic.cfg",
        "custom_doom_defend_the_center": "defend_the_center.cfg",
        "custom_doom_deadly_corridor": "deadly_corridor.cfg",
        "custom_doom_health_gathering": "health_gathering_supreme.cfg",
    }
    
    if env_name not in scenarios:
        raise ValueError(f"Unknown env name: {env_name}")
    
    scenario_file = scenarios[env_name]

    # --- 资源检查逻辑 ---
    def _ensure_game_variables_in_cfg(scenario_name):
        local_dir = os.path.join(os.path.dirname(__file__), '..', '_vizdoom')
        local_dir = os.path.abspath(local_dir)
        os.makedirs(local_dir, exist_ok=True)
        local_cfg_path = os.path.join(local_dir, scenario_name)
        
        # 检查 WAD 是否存在
        wad_name = scenario_name.replace('.cfg', '.wad')
        # 1. 检查本地 _vizdoom
        local_wad_path = os.path.join(local_dir, wad_name)
        # 2. 检查系统库路径
        sys_wad_path = os.path.join(vzd.scenarios_path, wad_name)
        
        final_wad_path = None
        if os.path.exists(local_wad_path):
            final_wad_path = local_wad_path
        elif os.path.exists(sys_wad_path):
            final_wad_path = sys_wad_path # 使用系统路径
        else:
            # 两个地方都没有，报错并提示
            raise FileNotFoundError(
                f"\n\n🛑 CRITICAL ERROR: WAD file '{wad_name}' not found!\n"
                f"Checked locations:\n  1. {local_wad_path}\n  2. {sys_wad_path}\n\n"
                f"👉 FIX: Run this command in container:\n"
                f"   cp /usr/local/lib/python3.10/site-packages/vizdoom/scenarios/{wad_name} src/_vizdoom/\n"
            )

        # 读取原始 CFG
        orig_cfg_path = os.path.join(vzd.scenarios_path, scenario_name)
        if not os.path.exists(orig_cfg_path):
             # 尝试从本地读
             if os.path.exists(local_cfg_path):
                 with open(local_cfg_path, 'r') as f: content = f.read()
             else:
                 return scenario_name # 放弃治疗，直接返回名字
        else:
            with open(orig_cfg_path, 'r') as f: content = f.read()

        # 暴力替换 WAD 路径为绝对路径或文件名（防止相对路径解析错误或重复拼接）
        import re
        # 如果 WAD 位于 local_dir，则只写入文件名（cfg 与 wad 同目录），避免引擎将 cfg 目录与绝对路径拼接导致重复路径
        if str(final_wad_path).startswith(str(local_dir)):
            content = re.sub(r'doom_scenario_path\s*=\s*.*', f'doom_scenario_path = {wad_name}', content)
        else:
            content = re.sub(r'doom_scenario_path\s*=\s*.*', f'doom_scenario_path = {final_wad_path}', content)

        # 确保变量存在：如果已有 available_game_variables，保证包含 HITCOUNT
        m = re.search(r'available_game_variables\s*=\s*\{([^}]*)\}', content)
        if m:
            vars_inner = m.group(1)
            vars_set = set(v.strip() for v in vars_inner.split())
            needed = {'KILLCOUNT', 'HITCOUNT', 'AMMO2', 'HEALTH', 'FRAGCOUNT'}
            if not needed.issubset(vars_set):
                merged = ' '.join(sorted(vars_set.union(needed)))
                content = re.sub(r'available_game_variables\s*=\s*\{[^}]*\}', f'available_game_variables = {{ {merged} }}', content)
        else:
            content += '\navailable_game_variables = { KILLCOUNT HITCOUNT AMMO2 HEALTH FRAGCOUNT }\n'
        
        # [新增] 物理外挂：加快转身速度，方便 AI 快速索敌
        if 'player_turn_speed' not in content:
            content += '\nplayer_turn_speed = 300\n'

        with open(local_cfg_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"[Info] Patched config: {local_cfg_path}")
        print(f"[Info] Using WAD: {final_wad_path}")
        return local_cfg_path

    scenario_file_path = _ensure_game_variables_in_cfg(scenario_file)

    # 2. 构造 Spec
    base_spec = get_spec_by_scenario(scenario_file)
    if base_spec is None:
        from gymnasium.spaces import Discrete
        # 默认给个 3 动作，后面会被 Wrapper 改掉
        env_spec = DoomSpec(env_name, scenario_file_path, Discrete(3)) 
    else:
        env_spec = DoomSpec(env_name, scenario_file_path, base_spec.action_space, base_spec.reward_scaling)

    # 3. 使用 make_doom_env_from_spec 创建基础环境（保持 SampleFactory 的兼容性）
    env = make_doom_env_from_spec(env_spec, env_name, cfg, env_config, render_mode=render_mode, **kwargs)

    # 尝试查找底层的 VizdoomEnv 实例，并对其进行小补丁：禁音 & 支持列表动作
    try:
        base = env
        while hasattr(base, 'env'):
            base = base.env
        # base 现在应该是 VizdoomEnv
        try:
            # 强力禁音
            if hasattr(base, 'game') and base.game is not None:
                base.game.set_sound_enabled(False)
                base.game.set_audio_buffer_enabled(False)
        except Exception:
            pass

        # monkey-patch _convert_actions，允许直接传入 list/tuple/numpy 的动作
        import types as _types
        if hasattr(base, '_convert_actions'):
            old_convert = base._convert_actions
            def _convert_actions_override(self, actions):
                if isinstance(actions, (list, tuple, np.ndarray)):
                    return actions
                return old_convert(actions)
            base._convert_actions = _types.MethodType(_convert_actions_override, base)
            print("[Info] Patched underlying VizdoomEnv to accept list/tuple actions and disabled audio.")
    except Exception as e:
        print(f"[Warning] Could not patch underlying VizdoomEnv: {e}")

    # 4. 依次套上 Wrapper (顺序很重要: 内 -> 外)
    # 先处理图像
    env = ImageCleaningWrapper(env)
    # 再处理奖励
    env = RewardShapingWrapper(env)
    # 最后处理动作 (最外层，因为它改变了 Action Space 的形状)
    env = CompositeActionWrapper(env)

    return env
