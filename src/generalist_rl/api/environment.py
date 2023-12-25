from typing import Dict, Union
import dataclasses
from generalist_rl.api.datatypes import Action

import torch
import numpy as np

@dataclasses.dataclass
class EnvStepResult:
    obs: Dict
    reward: Union[torch.Tensor, np.ndarray]
    done: Union[torch.Tensor, np.ndarray]
    info: Dict = None
    truncated: Union[torch.Tensor, np.ndarray] = None

class Environment:
    @property
    def num_envs(self) -> int:
        raise NotImplementedError()
    
    @property
    def num_obs(self) -> int:
        raise NotImplementedError()
    
    @property
    def num_privileged_obs(self) -> int:
        raise NotImplementedError() 

    @property
    def num_actions(self) -> int:
        raise NotImplementedError()
    
    def reset(self) -> EnvStepResult:
        raise NotImplementedError()

    def step(self, action: Action) -> EnvStepResult:
        raise NotImplementedError()
