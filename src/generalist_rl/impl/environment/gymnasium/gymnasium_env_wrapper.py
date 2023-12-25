# Wrapper for gymnasium environments
from generalist_rl.api.environment import Environment, EnvStepResult, Action
import numpy as np
import torch
from typing import Optional

import gymnasium as gym

class GymEnv(Environment):
    def __init__(self, env_name: str, num_envs:int, max_episode_steps: Optional[int] = None, autoreset: Optional[bool] = None, device=torch.device('cpu')) -> None:
        self._env = gym.vector.make(env_name, num_envs=num_envs, max_episode_steps=max_episode_steps, autoreset=autoreset) if num_envs != 1 \
        else gym.make(env_name, max_episode_steps=max_episode_steps, autoreset=autoreset)
        self.device = device
    
    @property
    def num_envs(self) -> int:
        return 1 if not isinstance(self._env, gym.vector.VectorEnv) else self._env.num_envs
    
    @property
    def num_obs(self) -> int:
        return self._env.observation_space.shape[-1]
    
    @property
    def num_privileged_obs(self) -> int:
        return 0
    
    @property
    def num_actions(self) -> int:
        action_space = self._env.single_action_space if self.num_envs != 1 else self._env.action_space
        if isinstance(action_space, gym.spaces.Box):
            return action_space.shape[-1]
        if isinstance(action_space, gym.spaces.Discrete):
            return action_space.n

    def __get_obs_dict(self, obs: np.ndarray):
        return {
            "obs": torch.from_numpy(obs).float().to(self.device),
            "critic_obs": torch.from_numpy(obs).float().to(self.device),
        }
        
    
    def reset(self) -> EnvStepResult:
        obs, info = self._env.reset()
        return EnvStepResult(obs=self.__get_obs_dict(obs), reward=None, done=None, truncated=None, info=info)
    
    def step(self, action: Action) -> EnvStepResult:
        """Take a step in the environment.
        
        For vectorized environments, will step all environments in parallel.
        """
        action_np = action.x.squeeze().cpu().numpy()
        obs, reward, done, truncation, info = self._env.step(action_np)
        return EnvStepResult(
            obs=self.__get_obs_dict(obs), 
            reward=torch.tensor(reward, device=self.device).unsqueeze(-1), 
            done=torch.tensor(done, device=self.device).unsqueeze(-1), 
            truncated=torch.tensor(truncation, device=self.device).unsqueeze(-1), 
            info=info)
