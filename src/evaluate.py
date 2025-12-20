#!/usr/bin/env python3
import sys
import os
import argparse
import numpy as np
import torch
import gymnasium as gym
import cv2
from pathlib import Path

# --- 1. 环境路径设置 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- 2. 导入自定义模块 ---
import src.envs
from src.models.custom_model import make_vizdoom_actor_critic
from src.envs.vizdoom_env import create_vizdoom_env

# --- 3. PyTorch 安全补丁 ---
import numpy.dtypes
torch.serialization.add_safe_globals([
    np.core.multiarray.scalar, 
    np.dtype, 
    np.dtypes.Float64DType, 
    np.dtypes.Int64DType
])

# --- 4. 辅助类 ---
class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class DictObservationWrapper(gym.ObservationWrapper):
    """
    将 Box 观察空间包装为 Dict 观察空间，适配 Sample Factory。
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            "obs": env.observation_space
        })
    
    def observation(self, obs):
        return {"obs": obs}

def get_eval_config(args):
    """
    构造与训练时一致的配置对象。
    """
    return AttrDict(
        # 核心模型参数
        use_rnn=True,
        rnn_size=512,
        rnn_type='gru',
        actor_critic_share_weights=True,
        
        # 补全缺失的 Decoder 参数
        decoder_mlp_layers=[512],  # Sample Factory 默认通常是 [512]
        
        # 环境配置
        res_w=128,
        res_h=72,
        wide_aspect_ratio=False,
        env_frameskip=4,
        pixel_format='CHW',
        
        # 归一化参数 (Sample Factory ActorCritic 初始化必需)
        normalize_input=True,
        normalize_input_keys=None,
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        normalize_returns=True,
        
        # 补全其他可能需要的默认参数
        nonlinearity='relu',
        use_encoder_linear=True,
    )

def main():
    parser = argparse.ArgumentParser(description="VizDoom Evaluation Script (Whitebox)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--env", type=str, default="custom_doom_health_gathering", help="Env name")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--video-dir", type=str, default="dist/final_videos", help="Output folder")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    
    args = parser.parse_args()
    device = torch.device(args.device)

    # 🔍 尝试从 checkpoint 自动推断 rnn_size，以确保评估时模型结构与训练时一致
    inferred_rnn_size = None
    try:
        tmp_ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        tmp_state = tmp_ckpt['model'] if 'model' in tmp_ckpt else tmp_ckpt
        for k, v in tmp_state.items():
            # GRU 的 weight_ih_l0 形状为 (3 * hidden_size, input_size)
            if k.endswith('.gru.weight_ih_l0') or '.gru.weight_ih_l0' in k:
                size = v.shape[0]
                if size % 3 == 0:
                    inferred_rnn_size = size // 3
                    break
    except Exception:
        inferred_rnn_size = None

    # 🔍 尝试从 checkpoint 自动推断 decoder 输出大小（action/value head 的输入维度）
    inferred_decoder_out = None
    try:
        tmp_ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        tmp_state = tmp_ckpt['model'] if 'model' in tmp_ckpt else tmp_ckpt
        for k, v in tmp_state.items():
            if k.endswith('action_heads.linear.weight') or '.action_heads.linear.weight' in k:
                inferred_decoder_out = v.shape[1]
                break
            if k.endswith('value_head.weight') or '.value_head.weight' in k:
                inferred_decoder_out = v.shape[1]
                break
    except Exception:
        inferred_decoder_out = None

    cfg = get_eval_config(args)
    if inferred_rnn_size is not None and cfg.rnn_size != inferred_rnn_size:
        print(f"[Info] Inferred rnn_size={inferred_rnn_size} from checkpoint; overriding cfg.rnn_size (was {cfg.rnn_size})")
        cfg.rnn_size = inferred_rnn_size
    if inferred_decoder_out is not None:
        if not isinstance(cfg.decoder_mlp_layers, list) or cfg.decoder_mlp_layers[-1] != inferred_decoder_out:
            print(f"[Info] Inferred decoder_out_size={inferred_decoder_out} from checkpoint; overriding cfg.decoder_mlp_layers (was {cfg.decoder_mlp_layers})")
            cfg.decoder_mlp_layers = [inferred_decoder_out]

    print(f"\n🎬 === Starting Evaluation ===")
    print(f"   Env:        {args.env}")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Device:     {device}")
    
    # 1. 创建环境
    print("   Creating environment...")
    try:
        raw_env = create_vizdoom_env(args.env, cfg=cfg, render_mode='rgb_array')
    except Exception as e:
        print(f"⚠️  Env creation fallback: {e}")
        raw_env = create_vizdoom_env(args.env, render_mode='rgb_array')

    # 2. 包装 Dict 空间
    if not isinstance(raw_env.observation_space, gym.spaces.Dict):
        print("   Wrapping environment in DictObservationWrapper...")
        raw_env = DictObservationWrapper(raw_env)

    # 3. 准备视频保存路径
    env = raw_env
    video_path = os.path.abspath(args.video_dir)
    os.makedirs(video_path, exist_ok=True)
    
    print(f"   Obs Space: {env.observation_space}")
    print(f"   Act Space: {env.action_space}")

    # 4. 初始化模型
    print("🧠 Initializing model architecture...")
    model = make_vizdoom_actor_critic(cfg, env.observation_space, env.action_space)
    model.to(device)
    model.eval()

    # 5. 加载权重
    print("📥 Loading weights...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("✅ Weights loaded.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        sys.exit(1)

    # 6. 评估循环
    print("\n🚀 Starting Run Loop...")
    rewards = []
    
    for i in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        
        # 准备视频写入器
        video_name = os.path.join(video_path, f"{args.env}-ep{i}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = None

        # 初始化 RNN (Batch=1)
        rnn_states = torch.zeros(1, cfg.rnn_size, device=device)

        while not done:
            # 捕获当前画面
            frame = env.render()
            if frame is not None:
                if video_writer is None:
                    h, w, _ = frame.shape
                    video_writer = cv2.VideoWriter(video_name, fourcc, 30, (w, h))
                # ViZDoom 返回 RGB，OpenCV 需要 BGR
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            step += 1
            
            # 数据预处理
            if isinstance(obs, dict):
                obs_data = obs['obs']
            else:
                obs_data = obs
            
            # 转为 Tensor [1, 1, 84, 84]
            obs_tensor = torch.from_numpy(obs_data).float().to(device).unsqueeze(0)
            obs_dict = {'obs': obs_tensor}

            # 推理
            with torch.no_grad():
                result = model(obs_dict, rnn_states, values_only=False)
            
            # 兼容不同的 Key 返回
            action_logits = result.get('action_logits', result.get('logits'))
            rnn_states = result['new_rnn_states']
            
            # 动作采样
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample().item()
            
            # 步进
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            done = terminated or truncated

        if video_writer:
            video_writer.release()
        rewards.append(ep_reward)
        print(f"   Episode {i+1}: Reward = {ep_reward:.2f}, Steps = {step}")

    env.close()
    print(f"\n📊 Result: Average Reward = {np.mean(rewards):.2f}")
    print(f"💾 Videos saved at: {video_path}")

if __name__ == "__main__":
    main()