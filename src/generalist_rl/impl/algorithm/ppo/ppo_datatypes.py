from generalist_rl.api.datatypes import AnalyzedResult

import torch


class PPORolloutAnalyzedResult(AnalyzedResult):
    def __init__(
        self,
        action_logprobs: torch.Tensor,
        state_values: torch.Tensor,
        returns: torch.Tensor = None,
        advantages: torch.Tensor = None,
    ):
        super().__init__(
            action_logprobs=action_logprobs,
            state_values=state_values,
            returns=returns,
            advantages=advantages,
        )


class PPOTrainerAnalyzedResult(AnalyzedResult):
    def __init__(
        self,
        action_logprobs: torch.Tensor,
        state_values: torch.Tensor,
        policy_entropy: torch.Tensor,
    ):
        super().__init__(
            action_logprobs=action_logprobs,
            state_values=state_values,
            policy_entropy=policy_entropy,
        )
