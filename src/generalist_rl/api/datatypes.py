import dataclasses
from typing import Dict, Union

import torch
import numpy as np

# actor to policy
class PolicyState:
    pass


@dataclasses.dataclass
class RolloutRequest:
    obs: Dict
    policy_state: PolicyState = None

# policy to actor
@dataclasses.dataclass
class Action:
    x: Union[torch.Tensor, np.ndarray]


class AnalyzedResult:
    pass


@dataclasses.dataclass
class RolloutResult:
    action: Action
    policy_state: PolicyState = None
    analyzed_result: AnalyzedResult = None

# actor to trainer
@dataclasses.dataclass
class SampleBatch:
    obs: Dict = None
    policy_state: PolicyState = None
    
    action: Union[torch.Tensor, np.ndarray] = None
    analyzed_result: AnalyzedResult = None

    reward: Union[torch.Tensor, np.ndarray] = None
    done: Union[torch.Tensor, np.ndarray] = None
    
    truncated: Union[torch.Tensor, np.ndarray] = None
    
    def __post_init__(self):
        if self.truncated is None:
            self.truncated = torch.zeros(1, dtype=torch.bool)
