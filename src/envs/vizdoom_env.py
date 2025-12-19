import os
import vizdoom as vzd
import gymnasium as gym
import numpy as np
from sample_factory.envs.env_utils import register_env
from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec, DoomSpec, DOOM_ENVS
from src.envs.wrappers import RewardShapingWrapper, ImageCleaningWrapper

def get_spec_by_scenario(scenario_name):
    """从内置的 DOOM_ENVS 中找到匹配场景文件的规格"""
    for spec in DOOM_ENVS:
        if spec.env_spec_file == scenario_name:
            return spec
    return None

def create_vizdoom_env(env_name, cfg=None, env_config=None, **kwargs):
    """
    基础环境创建函数。
    """
    # 1. 定义场景映射
    scenarios = {
        "custom_doom_basic": "basic.cfg",
        "custom_doom_defend_the_center": "defend_the_center.cfg",
        "custom_doom_deadly_corridor": "deadly_corridor.cfg",
        "custom_doom_health_gathering": "health_gathering.cfg",
    }
    
    if env_name not in scenarios:
        raise ValueError(f"Unknown env name: {env_name}")
    
    scenario_file = scenarios[env_name]
    
    # 2. 构造 DoomSpec 对象
    # 我们尝试从内置规格中复制 action_space 等信息
    base_spec = get_spec_by_scenario(scenario_file)
    
    if base_spec is not None:
        # 创建一个新的 Spec，但名字改为我们的自定义名字
        env_spec = DoomSpec(
            env_name, 
            scenario_file, 
            base_spec.action_space, # 沿用原始的动作空间定义
            reward_scaling=base_spec.reward_scaling,
            num_agents=base_spec.num_agents
        )
    else:
        # 如果没找到（理论上不会），使用保守的默认值
        from gymnasium.spaces import Discrete
        env_spec = DoomSpec(env_name, scenario_file, Discrete(3))

    # 3. 使用 Sample Factory 的基础工具创建环境
    # 传入 spec, _env_name, cfg, env_config
    env = make_doom_env_from_spec(env_spec, env_name, cfg, env_config, **kwargs)
    
    # 4. 添加自定义 Wrapper
    env = RewardShapingWrapper(env)
    
    return env
