from generalist_rl.api.datatypes import SampleBatch
from .ppo_datatypes import PPORolloutAnalyzedResult

import torch

def compute_returns_gae(samples: SampleBatch, gamma=0.99, lam=0.95):
    # TODO: consider using state value as proxy reward for truncated transitions
    rewards = samples.reward
    dones = samples.done
    # truncated = samples.truncated
    
    rollout_analyzed_result: PPORolloutAnalyzedResult = samples.analyzed_result
    state_values = rollout_analyzed_result.state_values

    # if this state is done, we do not add gamma * next_state_value
    delta = rewards[:-1] + gamma * state_values[1:] * torch.logical_not(dones[:-1]) - state_values[:-1]
    # if this state is truncated or done, we do not add advantage[t + 1]
    m = gamma * lam * torch.logical_not(dones[:-1])
    
    advantages = torch.zeros_like(rewards)
    for t in reversed(range(rewards.size(0) - 1)):
        advantages[t] = delta[t] + m[t] * advantages[t + 1]

    returns = advantages + state_values
    advantages[:-1] = (advantages[:-1] - advantages[:-1].mean(dim=0, keepdim=True)) / (advantages[:-1].std(dim=0, keepdim=True) + 1e-8)
    return returns, advantages
        