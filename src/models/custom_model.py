import torch
from torch import nn
import gymnasium as gym
from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import MlpDecoder
from sample_factory.model.core import ModelCore, ModelCoreRNN
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.utils.utils import log
# 修复：在 SF 2.1.1 中，类名通常是 ActionParameterizationDefault
from sample_factory.model.action_parameterization import ActionParameterizationDefault

class CustomVizdoomEncoder(Encoder):
    """
    手写 CNN Encoder (NatureCNN 风格)。
    输入: (1, 84, 84)
    输出: 512 维特征向量
    """
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        
        # 修复：兼容 Dict 空间和 Box 空间
        if isinstance(obs_space, gym.spaces.Dict):
            self.obs_shape = obs_space["obs"].shape
        else:
            self.obs_shape = obs_space.shape
            
        input_channels = self.obs_shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.obs_shape)
            n_flatten = self.cnn(dummy_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU()
        )
        
        self.encoder_out_size = 512

    def forward(self, obs_dict):
        obs = obs_dict["obs"]
        
        # 处理 5D 输入 [T, B, C, H, W] (RNN 训练时常见)
        if obs.dim() == 5:
            T, B, C, H, W = obs.shape
            obs = obs.reshape(T * B, C, H, W)
            features = self.cnn(obs)
            out = self.fc(features)
            return out.reshape(T, B, -1)
            
        features = self.cnn(obs)
        return self.fc(features)

    def get_out_size(self) -> int:
        return self.encoder_out_size

class CustomLSTMCore(ModelCoreRNN):
    """
    手写 LSTM Core。
    接管特征向量，输出带有记忆的特征。
    """
    def __init__(self, cfg, input_size):
        super().__init__(cfg, input_size)
        self.lstm = nn.LSTM(input_size, cfg.hidden_size, batch_first=False)
        self.core_output_size = cfg.hidden_size

    def forward(self, head_output, rnn_states):
        # head_output shape: [T, B, 512]
        # rnn_states shape: [B, 2 * hidden_size] (SF 2.x 默认将 h 和 c 拼接在 dim 1)
        
        # LSTM 需要 (h, c) 元组，且形状为 [num_layers, B, H]
        h, c = torch.split(rnn_states, self.cfg.hidden_size, dim=1)
        h = h.unsqueeze(0).contiguous() # [1, B, H]
        c = c.unsqueeze(0).contiguous()
        
        output, (h_n, c_n) = self.lstm(head_output, (h, c))
        
        # 将新的 h 和 c 拼接回 [B, 2 * H] 返回给 SF
        new_rnn_states = torch.cat([h_n.squeeze(0), c_n.squeeze(0)], dim=1)
        
        return output, new_rnn_states

    def get_out_size(self) -> int:
        return self.core_output_size

class CustomVizdoomActorCritic(ActorCritic):
    """
    继承 ActorCritic 并显式初始化所有组件。
    """
    def __init__(self, cfg, obs_space, action_space):
        # 1. 先调用基类初始化
        super().__init__(obs_space, action_space, cfg)
        
        # 2. 显式保存属性
        self.cfg = cfg
        self.obs_space = obs_space
        self.action_space = action_space
        
        # 3. 显式创建核心组件
        self.encoder = CustomVizdoomEncoder(cfg, obs_space)
        
        if cfg.use_rnn:
            self.core = CustomLSTMCore(cfg, self.encoder.get_out_size())
        else:
            self.core = ModelCore(cfg, self.encoder.get_out_size())
            
        self.decoder = MlpDecoder(cfg, self.core.get_out_size())
        
        # 4. 修复：ActionParameterizationDefault 本身就是 nn.Module
        self.action_parameterization = ActionParameterizationDefault(
            cfg, self.decoder.get_out_size(), action_space
        )
        # 直接使用对象本身作为动作头
        self.action_heads = self.action_parameterization
        
        # 值头
        self.value_head = nn.Linear(self.decoder.get_out_size(), 1)

    def forward(self, obs_dict, rnn_states, values_only=False):
        """
        实现前向传播逻辑。
        """
        # 1. 观察归一化
        normalized_obs = self.normalize_obs(obs_dict)
        
        # 2. 编码器
        x = self.encoder(normalized_obs)
        
        # 3. 核心层 (处理 RNN 维度)
        if self.cfg.use_rnn:
            if x.dim() == 2:
                x = x.unsqueeze(0)
                x, new_rnn_states = self.core(x, rnn_states)
                x = x.squeeze(0)
            else:
                x, new_rnn_states = self.core(x, rnn_states)
        else:
            x, new_rnn_states = self.core(x, rnn_states)
            
        # 4. 解码器
        decoder_out = self.decoder(x)
        
        # 5. 动作头和值头
        result = {"new_rnn_states": new_rnn_states}
        
        # 计算值函数
        result["values"] = self.value_head(decoder_out).squeeze(-1)
        
        if not values_only:
            # 计算动作逻辑值
            # action_parameterization 会处理 Discrete/Box 等不同空间的输出
            result["action_logits"] = self.action_heads(decoder_out)
                
        return result

def make_vizdoom_actor_critic(cfg, obs_space, action_space):
    """
    工厂函数：现在只需实例化自定义的 ActorCritic 类。
    """
    return CustomVizdoomActorCritic(cfg, obs_space, action_space)

def register_models():
    """
    将自定义模型工厂注册到 Sample Factory。
    """
    global_model_factory().register_actor_critic_factory(make_vizdoom_actor_critic)
    log.info("Successfully registered custom_vizdoom_model via global_model_factory")

