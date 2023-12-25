from generalist_rl.api.environment import Environment, EnvStepResult, Action
import torch

from legged_gym.utils import task_registry

class LeggedGymEnv(Environment):
    def __init__(self, args) -> None:
        self._env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    @property
    def num_envs(self) -> int:
        return self._env.num_envs
    
    @property
    def num_obs(self) -> int:
        return self._env.num_obs
    
    @property
    def num_privileged_obs(self) -> int:
        return self._env.num_privileged_obs if self._env.num_privileged_obs is not None else 0
    
    @property
    def num_actions(self) -> int:
        return self._env.num_actions
    
    def reset(self) -> EnvStepResult:
        """Only need to call once upon initialization, will automatically reset if the last episode was done in step(). Upon reset, will return done=True and the observation of the start of the next episode.
        
        For vectorized environments, will reset the environments that are done. 
        For environments that are reset, will return done=True and the observation of the start of the next episode.
        """
        obs, privileged_obs = self._env.reset()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs_dict = {
            "obs": obs,
            "critic_obs": critic_obs,
        }
        return EnvStepResult(obs=obs_dict, reward=None, done=torch.ones(self.num_envs, dtype=torch.bool), info=None)
    
    def step(self, action: Action) -> EnvStepResult:
        """Take a step in the environment.
        
        For vectorized environments, will step all environments in parallel.
        """
        obs, privileged_obs, reward, done, extras = self._env.step(action.x)
        truncated = extras.pop('time_outs').unsqueeze(-1) if 'time_outs' in extras else None
        obs_dict = {
            "obs": obs,
            "critic_obs": privileged_obs,
        }
        return EnvStepResult(obs=obs_dict, reward=reward.unsqueeze(-1), done=done.unsqueeze(-1), info=extras, truncated=truncated)
