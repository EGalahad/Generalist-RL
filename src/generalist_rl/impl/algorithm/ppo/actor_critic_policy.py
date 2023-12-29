from generalist_rl.api.datatypes import RolloutRequest, RolloutResult, SampleBatch, Action
from generalist_rl.api.policy import SingleModelPytorchPolicy

from .ppo_datatypes import PPORolloutAnalyzedResult, PPOTrainerAnalyzedResult

import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal, Normal

class ActorCriticNet(nn.Module):
    def __init__(self, num_obs, num_actions, has_continuous_action_space, action_std_init=0.6):
        super(ActorCriticNet, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = num_actions
            self.action_var = nn.Parameter(torch.full((num_actions,), action_std_init * action_std_init))
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                            nn.Linear(num_obs, 128),
                            nn.ELU(),
                            nn.Linear(128, 64),
                            nn.ELU(),
                            nn.Linear(64, 32),
                            nn.ELU(),
                            nn.Linear(32, num_actions),
                            # nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(num_obs, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, num_actions),
                            nn.Softmax(dim=-1)
                        )
        self.distribution = None

        # critic
        self.critic = nn.Sequential(nn.Linear(num_obs, 128),
                        nn.ELU(),
                        nn.Linear(128, 64),
                        nn.ELU(),
                        nn.Linear(64, 32),
                        nn.ELU(),
                        nn.Linear(32, 1)
                )

        
    def act(self, obs):
        self.__update_distribution(obs)

        actions = self.distribution.sample()
        action_logprobs = self.distribution.log_prob(actions)
        state_values = self.critic(obs)

        # if not self.has_continuous_action_space:
        #     actions = actions.unsqueeze(-1)

        return actions.detach(), action_logprobs.unsqueeze(-1).detach(), state_values.detach()

    def evaluate(self, obs, action):
        self.__update_distribution(obs)
        
        action_logprobs = self.distribution.log_prob(action)
        policy_entropy = self.distribution.entropy()
        state_values = self.critic(obs)
        
        return action_logprobs.unsqueeze(-1), state_values, policy_entropy.unsqueeze(-1)

    def __update_distribution(self, obs: torch.Tensor):
        if self.has_continuous_action_space:
            action_mean = self.actor(obs)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            self.distribution = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(obs)
            self.distribution = Categorical(action_probs)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
      

class ActorCriticNet(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticNet, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


class ActorCriticPolicy(SingleModelPytorchPolicy):
    net: ActorCriticNet
    def __init__(self, num_obs, num_actions, has_continuous_action_space, action_std_init=None, device=None):
        self._device = device
        net = ActorCriticNet(num_obs, num_obs, num_actions, init_noise_std=action_std_init)
        super(ActorCriticPolicy, self).__init__(net)
    
    @property
    def default_policy_state(self):
        return None

    def rollout(self, requests: RolloutRequest) -> RolloutResult:
        with torch.inference_mode():
            # actions, action_logprobs, state_values = self.net.act(requests.obs['obs'])
            actions = self.net.act(requests.obs['obs'])
            action_logprobs = self.net.get_actions_log_prob(actions)
            state_values = self.net.evaluate(requests.obs['obs'])
        
        return RolloutResult(
            action=Action(x=actions),
            analyzed_result=PPORolloutAnalyzedResult(
                action_logprobs=action_logprobs,
                state_values=state_values
            )
        )
    
    def analyze(self, sample: SampleBatch, **kwargs) -> PPOTrainerAnalyzedResult:
        # action_logprobs, state_values, policy_entropy = self.net.evaluate(sample.obs['obs'], sample.action['x'])
        self.net.update_distribution(sample.obs['obs'])
        action_logprobs = self.net.get_actions_log_prob(sample.action['x'])
        policy_entropy = self.net.distribution.entropy()
        state_values = self.net.evaluate(sample.obs['obs'])
        return PPOTrainerAnalyzedResult(
            action_logprobs=action_logprobs,
            state_values=state_values,
            policy_entropy=policy_entropy
        )
        
