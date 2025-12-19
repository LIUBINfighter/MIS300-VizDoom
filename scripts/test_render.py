
import gymnasium as gym
import torch
from src.envs.vizdoom_env import create_vizdoom_env
from src.models.custom_model import make_vizdoom_actor_critic
from src.evaluate import AttrDict, DictObservationWrapper
import numpy as np

def test_render():
    cfg = AttrDict(
        use_rnn=True,
        rnn_size=512,
        rnn_type='gru',
        actor_critic_share_weights=True,
        decoder_mlp_layers=[512],
        res_w=128,
        res_h=72,
        wide_aspect_ratio=False,
        env_frameskip=4,
        pixel_format='CHW',
        normalize_input=True,
        normalize_input_keys=None,
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        normalize_returns=True,
        nonlinearity='relu',
        use_encoder_linear=True,
    )
    
    env_name = "custom_doom_health_gathering"
    print(f"Creating env {env_name}...")
    env = create_vizdoom_env(env_name, cfg=cfg, render_mode='rgb_array')
    
    print("Resetting env...")
    obs, info = env.reset()
    
    print(f"Obs shape: {obs.shape}")
    
    print("Calling render()...")
    frame = env.render()
    if frame is None:
        print("❌ render() returned None!")
    else:
        print(f"✅ render() returned frame with shape {frame.shape}")
        
    print("Closing env...")
    env.close()

if __name__ == "__main__":
    test_render()
