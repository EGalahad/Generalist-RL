from generalist_rl.api.datatypes import RolloutRequest, RolloutResult, SampleBatch, Action
from generalist_rl.api.policy import SingleModelPytorchPolicy

from .ppo_datatypes import PPORolloutAnalyzedResult, PPOTrainerAnalyzedResult

import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal

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


class ActorCriticPolicy(SingleModelPytorchPolicy):
    net: ActorCriticNet
    def __init__(self, num_obs, num_actions, has_continuous_action_space, action_std_init=None, device=None):
        self._device = device
        net = ActorCriticNet(num_obs, num_actions, has_continuous_action_space, action_std_init)
        super(ActorCriticPolicy, self).__init__(net)
    
    @property
    def default_policy_state(self):
        return None

    def rollout(self, requests: RolloutRequest) -> RolloutResult:
        with torch.inference_mode():
            actions, action_logprobs, state_values = self.net.act(requests.obs['obs'])
        
        return RolloutResult(
            action=Action(x=actions),
            analyzed_result=PPORolloutAnalyzedResult(
                action_logprobs=action_logprobs,
                state_values=state_values
            )
        )
    
    def analyze(self, sample: SampleBatch, **kwargs) -> PPOTrainerAnalyzedResult:
        action_logprobs, state_values, policy_entropy = self.net.evaluate(sample.obs['obs'], sample.action['x'])
        return PPOTrainerAnalyzedResult(
            action_logprobs=action_logprobs,
            state_values=state_values,
            policy_entropy=policy_entropy
        )
        
