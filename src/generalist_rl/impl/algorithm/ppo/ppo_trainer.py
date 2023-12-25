from generalist_rl.api.datatypes import SampleBatch
from generalist_rl.api.trainer import PyTorchTrainer, TrainerStepResult

from .ppo_datatypes import PPORolloutAnalyzedResult
from .actor_critic_policy import ActorCriticPolicy
from .utils import compute_returns_gae

import torch
import torch.nn.functional as F


class PPOTrainer(PyTorchTrainer):
    policy: ActorCriticPolicy
    def __init__(self, policy: ActorCriticPolicy, lr_actor=3e-4, lr_critic=1e-3, weight_decay=0.0, **kwargs):
        super().__init__(policy)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.net.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.net.critic.parameters(), "lr": lr_critic}
            ]
        )
    
        self.ppo_epochs = kwargs.get("ppo_epochs", 80)
        self.ppo_clip = kwargs.get("ppo_clip", 0.2)
        self.critic_loss_weight = kwargs.get("critic_loss_weight", 0.5)
        self.entropy_loss_weight = kwargs.get("entropy_loss_weight", 0.01)
        self.gamma = kwargs.get("gamma", 0.99)
        self.lamda = kwargs.get("lamda", 0.95)
            
    def step(self, samples: SampleBatch) -> TrainerStepResult:
        rollout_analyzed_result: PPORolloutAnalyzedResult = samples.analyzed_result

        returns, advantages = compute_returns_gae(samples, gamma=self.gamma, lamda=self.lamda)
        # breakpoint()

        returns = returns.detach()
        advantages = advantages.detach()
        old_action_logprobs = rollout_analyzed_result.action_logprobs.detach()
        
        for ppo_epoch in range(self.ppo_epochs):
            trainer_analyzed_result = self.policy.analyze(samples)

            importance_ratio = torch.exp(trainer_analyzed_result.action_logprobs - old_action_logprobs)

            surrogate1 = importance_ratio * advantages
            surrogate2 = torch.clamp(importance_ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
            
            actor_loss = -torch.mean(torch.min(surrogate1, surrogate2))
            critic_loss = F.mse_loss(trainer_analyzed_result.state_values, returns)
            entropy_loss = -torch.mean(trainer_analyzed_result.policy_entropy)
            
            # first_done = torch.where(samples.done)[0][0]
            # print("returns: ", returns[:first_done+1, 0, 0])
            # print("state_values: ", trainer_analyzed_result.state_values[:first_done+1, 0, 0])
            
            loss = actor_loss + self.critic_loss_weight * critic_loss + self.entropy_loss_weight * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        loss_stats = {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy_loss": entropy_loss.item()
        }

        reward_stats = {
            "reward": torch.mean(samples.reward).item(),
        }

        self.policy.inc_version()
        return TrainerStepResult(
            stats={**loss_stats, **reward_stats},
            step=self.policy.version,
        )
            
        
        
        
        