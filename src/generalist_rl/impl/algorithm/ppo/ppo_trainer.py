from generalist_rl.api.datatypes import SampleBatch
from generalist_rl.api.trainer import PyTorchTrainer, TrainerStepResult

from .ppo_datatypes import PPORolloutAnalyzedResult
from .actor_critic_policy import ActorCriticPolicy
from .utils import compute_returns_gae

import torch
import torch.nn.functional as F

from generalist_rl.utils.training import minibatch_generator


class PPOTrainer(PyTorchTrainer):
    policy: ActorCriticPolicy
    def __init__(self, policy: ActorCriticPolicy, lr_actor=3e-4, lr_critic=1e-3, weight_decay=0.0, **kwargs):
        super().__init__(policy)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.net.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.net.critic.parameters(), "lr": lr_critic},
                # {"params": self.policy.net.action_var, "lr": lr_actor}
            ]
        )
        # if policy has action_var, add it to optimizer
        if hasattr(self.policy.net, "action_var"):
            self.optimizer.add_param_group({"params": self.policy.net.action_var, "lr": lr_actor})
    
        self.ppo_epochs = kwargs.get("ppo_epochs", 80)
        self.ppo_clip = kwargs.get("ppo_clip", 0.2)
        self.critic_loss_weight = kwargs.get("critic_loss_weight", 0.5)
        self.entropy_loss_weight = kwargs.get("entropy_loss_weight", 0.01)
        self.gamma = kwargs.get("gamma", 0.99)
        self.lamda = kwargs.get("lamda", 0.95)
        self.num_minibatch = kwargs.get("num_minibatch", 4)
        self.shuffle = kwargs.get("shuffle", True)
            
    def step(self, samples: SampleBatch) -> TrainerStepResult:
        samples = compute_returns_gae(samples, gamma=self.gamma, lamda=self.lamda)
        # breakpoint()

        losses = []
        critic_losses = []
        actor_losses = []
        entropy_losses = []
        gae_returns = []
        rewards = []
        values = []

        for sample in minibatch_generator(samples, self.ppo_epochs, self.num_minibatch, self.shuffle):
            rollout_analyzed_result: PPORolloutAnalyzedResult = sample.analyzed_result
            trainer_analyzed_result = self.policy.analyze(sample)

            importance_ratio = torch.exp(trainer_analyzed_result.action_logprobs - rollout_analyzed_result.action_logprobs.detach())

            surrogate1 = importance_ratio * rollout_analyzed_result.advantages.detach()
            surrogate2 = torch.clamp(importance_ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * rollout_analyzed_result.advantages.detach()
            
            actor_loss = -torch.mean(torch.min(surrogate1, surrogate2))
            critic_loss = F.mse_loss(trainer_analyzed_result.state_values, rollout_analyzed_result.returns.detach())
            entropy_loss = -torch.mean(trainer_analyzed_result.policy_entropy)
            
            loss = actor_loss + self.critic_loss_weight * critic_loss + self.entropy_loss_weight * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())
            entropy_losses.append(entropy_loss.item())
            rewards.append(torch.mean(sample.reward).item())
            gae_returns.append(torch.mean(rollout_analyzed_result.returns).item())
            values.append(torch.mean(trainer_analyzed_result.state_values).item())
            
        loss_stats = {
            "loss": torch.mean(torch.tensor(losses)).item(),
            "actor_loss": torch.mean(torch.tensor(actor_losses)).item(),
            "critic_loss": torch.mean(torch.tensor(critic_losses)).item(),
            "entropy_loss": torch.mean(torch.tensor(entropy_losses)).item(),
        }

        reward_stats = {
            "gae_return": torch.mean(torch.tensor(gae_returns)).item(),
            "reward": torch.mean(torch.tensor(rewards)).item(),
            "value": torch.mean(torch.tensor(values)).item(),
        }

        self.policy.inc_version()
        return TrainerStepResult(
            stats={**loss_stats, **reward_stats},
            step=self.policy.version,
        )
            
        
        
        
        