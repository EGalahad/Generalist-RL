import dataclasses
from typing import Dict, Union

from generalist_rl.utils.namedarray import NamedArray
import torch
import numpy as np


# actor to policy
class PolicyState:
    pass


@dataclasses.dataclass
class RolloutRequest:
    obs: NamedArray
    policy_state: PolicyState = None


# policy to actor
class Action(NamedArray):
    def __init__(self, x: Union[torch.Tensor, np.ndarray]):
        super(Action, self).__init__(x=x)


AnalyzedResult = NamedArray


class RolloutResult(NamedArray):
    def __init__(
        self,
        action: Action,
        analyzed_result: AnalyzedResult = None,
    ):
        super(RolloutResult, self).__init__(
            action=action,
            analyzed_result=analyzed_result,
        )

# actor to trainer
class SampleBatch(NamedArray):
    def __init__(
        self,
        obs: NamedArray = None,
        policy_state: PolicyState = None,
        action: Action = None,
        analyzed_result: AnalyzedResult = None,
        reward: Union[torch.Tensor, np.ndarray] = None,
        done: Union[torch.Tensor, np.ndarray] = None,
        truncated: Union[torch.Tensor, np.ndarray] = None,
    ):
        super(SampleBatch, self).__init__(
            obs=obs,
            policy_state=policy_state,
            action=action,
            analyzed_result=analyzed_result,
            reward=reward,
            done=done,
            truncated=truncated,
        )
