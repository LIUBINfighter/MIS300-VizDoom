import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
import gymnasium as gym
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.encoder import Encoder
from sample_factory.model.core import ModelCore, ModelCoreRNN
from sample_factory.model.decoder import MlpDecoder
from sample_factory.utils.utils import log
import torch.distributions as torch_d

# === 🚨 关键修复：通用型 Categorical ===
class CustomCategorical(torch_d.Categorical):
    def symmetric_kl_with_uniform_prior(self):
        """SF 需要的对称 KL 散度 (保持不变)"""
        probs = self.probs
        num_categories = probs.shape[-1]
        uniform_prob = 1.0 / num_categories
        kl_p_u = (probs * (probs.log() - torch.log(torch.tensor(uniform_prob, device=probs.device)))).sum(-1)
        kl_u_p = -torch.log(torch.tensor(uniform_prob, device=probs.device)) - (uniform_prob * probs.log()).sum(-1)
        return 0.5 * (kl_p_u + kl_u_p)

    def kl_divergence(self, other):
        """
        手动处理 KL 散度，绕过 PyTorch 的类型检查注册表。
        将 self 和 other 都视为标准的 torch.distributions.Categorical 进行计算。
        """
        # 1. 将自己(p)转为标准 Categorical
        p_dist = torch_d.Categorical(logits=self.logits)
        
        # 2. 将对方(q)转为标准 Categorical
        # 对方可能是 SampleFactory 的封装类，只要有 logits 属性即可
        if hasattr(other, 'logits'):
            q_dist = torch_d.Categorical(logits=other.logits)
        elif hasattr(other, 'probs'):
            q_dist = torch_d.Categorical(probs=other.probs)
        else:
            # 最后的保底，直接传过去试试
            q_dist = other
            
        # 3. 调用 PyTorch 官方实现的 KL(Categorical || Categorical)
        return torch_d.kl.kl_divergence(p_dist, q_dist)

# === 自定义动作头 ===
class CustomActionHead(nn.Module):
    def __init__(self, input_size, action_space):
        super().__init__()
        self.num_actions = action_space.n
        self.linear = nn.Linear(input_size, self.num_actions)

    def forward(self, x):
        return self.linear(x)

    def get_action_distribution(self, logits):
        return CustomCategorical(logits=logits)

class CustomVizdoomEncoder(Encoder):
    """手写 CNN Encoder"""
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        if isinstance(obs_space, gym.spaces.Dict):
            self.obs_shape = obs_space["obs"].shape
        else:
            self.obs_shape = obs_space.shape
        input_channels = self.obs_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.obs_shape)
            n_flatten = self.cnn(dummy_input).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flatten, 512), nn.ReLU())
        self.encoder_out_size = 512

    def forward(self, obs_dict):
        obs = obs_dict["obs"]
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

class CustomGRUCore(ModelCoreRNN):
    """手写 GRU Core"""
    def __init__(self, cfg, input_size):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.gru = nn.GRU(input_size, cfg.rnn_size, batch_first=False)
        self.core_output_size = cfg.rnn_size

    def forward(self, head_output, rnn_states):
        h = rnn_states.unsqueeze(0).contiguous()
        output, h_n = self.gru(head_output, h)
        new_rnn_states = h_n.squeeze(0)
        return output, new_rnn_states

    def get_out_size(self) -> int:
        return self.core_output_size

class CustomVizdoomActorCritic(ActorCritic):
    def __init__(self, cfg, obs_space, action_space):
        super().__init__(obs_space, action_space, cfg)
        self.cfg = cfg
        self.obs_space = obs_space
        self.action_space = action_space
        
        self.encoder = CustomVizdoomEncoder(cfg, obs_space)
        self.encoders = nn.ModuleList([self.encoder])
        
        if cfg.use_rnn:
            self.core = CustomGRUCore(cfg, self.encoder.get_out_size())
        else:
            self.core = ModelCore(cfg, self.encoder.get_out_size())
        self.cores = nn.ModuleList([self.core])
            
        self.decoder = MlpDecoder(cfg, self.core.get_out_size())
        
        self.action_heads = CustomActionHead(self.decoder.get_out_size(), action_space)
        self.value_head = nn.Linear(self.decoder.get_out_size(), 1)
        
        self.last_action_logits = None

    def forward_head(self, obs_dict):
        normalized_obs = self.normalize_obs(obs_dict)
        return self.encoder(normalized_obs)

    def forward_core(self, head_output, rnn_states):
        x = head_output
        if self.cfg.use_rnn:
            is_packed = isinstance(x, PackedSequence)
            if is_packed:
                x, new_rnn_states = self.core(x, rnn_states)
            elif x.dim() == 2:
                x = x.unsqueeze(0)
                x, new_rnn_states = self.core(x, rnn_states)
                x = x.squeeze(0)
            else:
                x, new_rnn_states = self.core(x, rnn_states)
        else:
            new_rnn_states = rnn_states
        return x, new_rnn_states

    def forward_tail(self, core_output, values_only=False, sample_actions=False):
        decoder_out = self.decoder(core_output)
        result = {}
        result["values"] = self.value_head(decoder_out).squeeze(-1)

        if not values_only:
            logits = self.action_heads(decoder_out)
            self.last_action_logits = logits

            result["logits"] = logits
            result["action_logits"] = logits

            if sample_actions:
                distribution = self.action_heads.get_action_distribution(logits)
                actions = distribution.sample()
                result["actions"] = actions
                result["log_prob_actions"] = distribution.log_prob(actions)
        
        return result

    def forward(self, obs_dict, rnn_states, values_only=False):
        x = self.forward_head(obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only=values_only, sample_actions=not values_only)
        result["new_rnn_states"] = new_rnn_states
        return result

    def get_rnn_size(self):
        return self.cfg.rnn_size

    def action_distribution(self, action_logits=None):
        if action_logits is None:
            action_logits = self.last_action_logits
        return self.action_heads.get_action_distribution(action_logits)

def make_vizdoom_actor_critic(cfg, obs_space, action_space):
    cfg.rnn_type = 'gru'
    return CustomVizdoomActorCritic(cfg, obs_space, action_space)

def register_models():
    global_model_factory().register_actor_critic_factory(make_vizdoom_actor_critic)
    log.info("Successfully registered custom_vizdoom_model (Final+CustomDist+UniversalKL) via global_model_factory")