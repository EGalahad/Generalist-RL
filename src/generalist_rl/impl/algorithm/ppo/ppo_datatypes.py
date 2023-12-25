from generalist_rl.api.datatypes import AnalyzedResult

import dataclasses
import torch

@dataclasses.dataclass
class PPORolloutAnalyzedResult(AnalyzedResult):
    # policy.rollout()
    action_logprobs: torch.Tensor
    state_values: torch.Tensor
    
    # # compute in trainer
    # advantages: torch.Tensor = None
    # returns: torch.Tensor = None


@dataclasses.dataclass
class PPOTrainerAnalyzedResult(AnalyzedResult):
    # policy.analyze()
    action_logprobs: torch.Tensor
    state_values: torch.Tensor
    policy_entropy: torch.Tensor
