from generalist_rl.api.datatypes import SampleBatch
from generalist_rl.api.buffer import Buffer

from .ppo_datatypes import PPORolloutAnalyzedResult

import torch
import numpy as np
from generalist_rl.utils.namedarray import recursive_aggregate

class BufferNamedArray(Buffer):
    def __init__(self, **kwargs):
        self.transitions = []
    
    def qsize(self) -> int:
        return len(self.transitions)
    
    def put(self, sample: SampleBatch):
        self.transitions.append(sample)
        
    def clear(self):
        del self.transitions[:]
        
    def get(self) -> SampleBatch:
        samples = recursive_aggregate(self.transitions, torch.stack)
        # samples.truncated[-1, ...] = True
        return samples


class PPOBufferTensorGPU(Buffer):
    def __init__(self, num_envs, num_transitions_per_env, num_obs, num_privileged_obs, num_action, device='cuda', has_continuous_action_space=True):
        self.obs = {
            "obs": torch.zeros((num_transitions_per_env, num_envs, num_obs), dtype=torch.float32, device=device),
            "critic_obs": None if num_privileged_obs == 0 else torch.zeros((num_transitions_per_env, num_envs, num_privileged_obs), dtype=torch.float32, device=device)
        }
        # self.policy_states = torch.zeros((num_transitions_per_env, num_envs, 1), dtype=torch.float32, device=device)

        self.actions = torch.zeros((num_transitions_per_env, num_envs, *((num_action,) if has_continuous_action_space else ())), dtype=torch.float32, device=device)
        self.analyzed_results = PPORolloutAnalyzedResult(
            action_logprobs=torch.zeros((num_transitions_per_env, num_envs, 1), dtype=torch.float32, device=device),
            state_values=torch.zeros((num_transitions_per_env, num_envs, 1), dtype=torch.float32, device=device),
        )
        
        self.rewards = torch.zeros((num_transitions_per_env, num_envs, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_transitions_per_env, num_envs, 1), dtype=torch.bool, device=device)
        self.truncated = torch.zeros((num_transitions_per_env, num_envs, 1), dtype=torch.bool, device=device)

        self.step = 0
        self.num_transitions_per_env = num_transitions_per_env
    
    @property
    def qsize(self) -> int:
        return self.num_transitions_per_env

    def put(self, sample: SampleBatch):
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError("Buffer is full")
        
        for key in self.obs.keys():
            if self.obs[key] is not None:
                self.obs[key][self.step] = sample.obs[key]
        # self.policy_states[self.step] = item.policy_state

        self.actions[self.step] = sample.action.x
        self.analyzed_results.action_logprobs[self.step] = sample.analyzed_result.action_logprobs
        self.analyzed_results.state_values[self.step] = sample.analyzed_result.state_values

        self.rewards[self.step] = sample.reward
        self.dones[self.step] = sample.done
        self.truncated[self.step] = sample.truncated
        
        self.step += 1
    
    def get(self) -> SampleBatch:
        """Returns the entire buffer."""
        return SampleBatch(
            obs=self.obs,
            policy_state=None,
            action=self.actions,
            analyzed_result=self.analyzed_results,
            reward=self.rewards,
            done=self.dones,
            truncated=self.truncated,
        )
    
    def clear(self):
        self.step = 0
        old_obs = self.obs
        self.obs = {}
        for key in old_obs.keys():
            if old_obs[key] is not None:
                self.obs[key] = torch.zeros_like(old_obs[key])
        del old_obs
        # self.policy_states = torch.zeros_like(self.policy_states)

        self.actions = torch.zeros_like(self.actions)
        self.analyzed_results = PPORolloutAnalyzedResult(
            action_logprobs=torch.zeros_like(self.analyzed_results.action_logprobs),
            state_values=torch.zeros_like(self.analyzed_results.state_values),
        )

        self.rewards = torch.zeros_like(self.rewards)
        self.dones = torch.zeros_like(self.dones)
        self.truncated = torch.zeros_like(self.truncated)
        

class PPOBufferTensorCPU(Buffer):
    def __init__(self, num_envs, num_transitions_per_env, num_obs, num_privileged_obs, num_action, device='cuda'):
        self.obs = {
            "obs": np.zeros((num_transitions_per_env, num_envs, num_obs), dtype=np.float32),
            "critic_obs": None if num_privileged_obs == 0 else np.zeros((num_transitions_per_env, num_envs, num_privileged_obs), dtype=np.float32)
        }
        # self.policy_states = np.zeros((num_transitions_per_env, num_envs, 1), dtype=np.float32)

        self.actions = np.zeros((num_transitions_per_env, num_envs, num_action), dtype=np.float32)
        self.analyzed_results = PPORolloutAnalyzedResult(
            action_logprobs=np.zeros((num_transitions_per_env, num_envs, 1), dtype=np.float32),
            state_values=np.zeros((num_transitions_per_env, num_envs, 1), dtype=np.float32),
        )
        
        self.rewards = np.zeros((num_transitions_per_env, num_envs, 1), dtype=np.float32)
        self.dones = np.zeros((num_transitions_per_env, num_envs, 1), dtype=np.bool)
        self.truncated = np.zeros((num_transitions_per_env, num_envs, 1), dtype=np.bool)

        self.step = 0
        self.num_transitions_per_env = num_transitions_per_env
        self.device = device
    
    @property
    def qsize(self) -> int:
        return self.num_transitions_per_env

    def put(self, sample: SampleBatch):
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError("Buffer is full")
        
        for key in self.obs.keys():
            if self.obs[key] is not None:
                self.obs[key][self.step] = sample.obs[key]
        # self.policy_states[self.step] = item.policy_state

        self.actions[self.step] = sample.action.x
        self.analyzed_results.action_logprobs[self.step] = sample.analyzed_result.action_logprobs
        self.analyzed_results.state_values[self.step] = sample.analyzed_result.state_values

        self.rewards[self.step] = sample.reward
        self.dones[self.step] = sample.done
        self.truncated[self.step] = sample.truncated
        
        self.step += 1
    
    def get(self) -> SampleBatch:
        """Returns the entire buffer."""
        return SampleBatch(
            obs=torch.from_numpy(self.obs, dtype=torch.float32).to(self.device),
            policy_state=None,
            action=torch.from_numpy(self.actions, dtype=torch.float32).to(self.device),
            analyzed_result=PPORolloutAnalyzedResult(
                action_logprobs=torch.from_numpy(self.analyzed_results.action_logprobs, dtype=torch.float32).to(self.device),
                state_values=torch.from_numpy(self.analyzed_results.state_values, dtype=torch.float32).to(self.device),
            ),
            reward=torch.from_numpy(self.rewards, dtype=torch.float32).to(self.device),
            done=torch.from_numpy(self.dones, dtype=torch.bool).to(self.device),
            truncated=torch.from_numpy(self.truncated, dtype=torch.bool).to(self.device),
        )
    
    def clear(self):
        self.step = 0
        old_obs = self.obs
        self.obs = {}
        for key in old_obs.keys():
            if old_obs[key] is not None:
                self.obs[key] = torch.zeros_like(old_obs[key])
        # self.policy_states = torch.zeros_like(self.policy_states)

        self.actions = torch.zeros_like(self.actions)
        self.analyzed_results = PPORolloutAnalyzedResult(
            action_logprobs=torch.zeros_like(self.analyzed_results.action_logprobs),
            state_values=torch.zeros_like(self.analyzed_results.state_values),
        )

        self.rewards = torch.zeros_like(self.rewards)
        self.dones = torch.zeros_like(self.dones)
        self.truncated = torch.zeros_like(self.truncated)
        