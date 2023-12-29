from generalist_rl.api.datatypes import SampleBatch
from .ppo_datatypes import PPORolloutAnalyzedResult

import torch


def compute_returns_gae(samples: SampleBatch, gamma=0.99, lamda=0.95):
    rollout_analyzed_result: PPORolloutAnalyzedResult = samples.analyzed_result
    state_values = rollout_analyzed_result.state_values

    # if the step is truncated, boostrap the value
    samples.reward += gamma * state_values * samples.truncated

    # if a step is truncated, or it is the last step collected, which may not be the end of an episode, we do not estimate its delta
    # since we need r_t + gamma V_{t+1} - V_t to estimate delta, we do not have V_{t+1} for the last step or truncated step
    # we only use its value to provide a baseline for the previous step
    samples = samples[:-1]
    rewards = samples.reward
    dones = samples.done
    truncated = samples.truncated
    dones |= truncated

    # if this state is done or truncated, we do not add gamma * next_state_value
    # the last step must be truncated, so we do not care its delta
    delta = (
        rewards
        + gamma * state_values[1:] * torch.logical_not(dones)
        - state_values[:-1]
    )
    # if this state is done or truncated, we do not add advantage[t + 1]
    m = gamma * lamda * torch.logical_not(dones)

    advantages = torch.zeros_like(rewards)
    advantage = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.size(0))):
        advantages[t] = advantage = delta[t] + m[t] * advantage

    # for done step, the advantage is r_t - V_t
    # for truncated step, the advantage is r_t - V_t, but we will not use it

    returns = advantages + state_values[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    samples.analyzed_result.returns = returns
    samples.analyzed_result.advantages = advantages
    return samples
