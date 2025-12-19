import torch
import numpy as np  # 添加导入
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
from sample_factory.model.actor_critic import create_actor_critic
from src.models import register_models
import gymnasium as gym

def test_model_registration():
    print("Testing model registration...")
    register_models()
    
    # 1. 修复：不要把 --model 放在 argv 中，因为它不是 SF 的标准命令行参数
    test_args = ["--env=custom_doom_health_gathering"]
    parser, cfg = parse_sf_args(argv=test_args, evaluation=True)
    
    # 2. 解析完整配置（这会填充默认值，如 obs_subtract_mean）
    cfg = parse_full_cfg(parser, argv=test_args)
    
    # 3. 手动注入自定义模型标识和必要参数
    cfg.model = "custom_vizdoom_model"
    cfg.use_rnn = True
    cfg.rnn_size = 128 # 修复：hidden_size -> rnn_size
    
    # 4. 模拟观察空间和动作空间
    # 修复：SF 2.x 默认期望 Dict 类型的观察空间
    obs_space = gym.spaces.Dict({
        "obs": gym.spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
    })
    action_space = gym.spaces.Discrete(3)
    
    print(f"Config initialized. Model: {cfg.model}, RNN: {cfg.use_rnn}")
    
    try:
        # 5. 创建模型
        model = create_actor_critic(cfg, obs_space, action_space)
        print("✓ Model created successfully!")
        
        # 6. 测试前向传播
        # SF 2.x 的输入通常是字典，且包含 Batch 维度
        obs = torch.randn(1, 1, 84, 84) # [Batch, Channels, H, W]
        obs_dict = {"obs": obs}
        
        # RNN 状态：SF 2.x 默认将 h 和 c 拼接，大小为 2 * rnn_size
        rnn_states = torch.zeros(1, 2 * cfg.rnn_size) # 修复：hidden_size -> rnn_size
        
        # 执行前向传播
        # ActorCritic 的 forward 返回一个包含 action_logits, values 等的 dict
        output = model(obs_dict, rnn_states)
        print("✓ Forward pass successful!")
        
        print("✓ Phase 2 Verification Successful!")
    except Exception as e:
        print(f"Error during model testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_registration()
