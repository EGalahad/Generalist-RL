from generalist_rl.api.datatypes import SampleBatch
from generalist_rl.api.policy import Policy

import dataclasses
from typing import Dict, Optional
import numpy as np


@dataclasses.dataclass
class TrainerStepResult:
    stats: Dict  # Stats to be logged.
    step: int  # current step count of trainer.
    agree_pushing: Optional[bool] = True  # whether agree to push parameters
    priorities: Optional[
        np.ndarray] = None  # New priorities of the PER buffer.


class Trainer:
    @property
    def policy(self) -> Policy:
        raise NotImplementedError()

    def step(self, samples: SampleBatch) -> TrainerStepResult:
        """Advances one training step given samples collected by actor workers.

        Example code:
          ...
          some_data = self.policy.analyze(sample)
          loss = loss_fn(some_data, sample)
          self.optimizer.zero_grad()
          loss.backward()
          ...
          self.optimizer.step()
          ...

        Args:
            samples (SampleBatch): A batch of data required for training.

        Returns:
            TrainerStepResult: Entry to be logged by trainer worker.
        """
        raise NotImplementedError()

    def distributed(self, **kwargs):
        """Make the trainer distributed."""
        raise NotImplementedError()

    def get_checkpoint(self, *args, **kwargs):
        """Get checkpoint of the model, which typically includes:
        1. Policy weights (e.g. neural network parameter).
        2. Optimizer state.
        Return:
            checkpoint to be saved.
        """
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint, **kwargs):
        raise NotImplementedError()


class PyTorchTrainer(Trainer):
    """PyTorch trainer base class.

    Provide distributed training."""

    def __init__(self, policy: Policy):
        self._policy = policy
        self._steps = 0

    @property
    def policy(self) -> Policy:
        return self._policy

    # def distributed(self, world_size: int, rank: int, init_method: str, **kwargs):
    #     """Make the trainer distributed."""
    #     is_gpu_process = all(
    #         [
    #             torch.cuda.is_available(),
    #             dist.is_nccl_available(),
    #             self.policy.device != torch.device("cpu"),
    #         ]
    #     )
    #     dist.init_process_group(
    #         backend="nccl" if is_gpu_process else "gloo",
    #         init_method=init_method,
    #         world_size=world_size,
    #         rank=rank,
    #     )
    #     if dist.is_initialized():
    #         logger.debug(
    #             f"Trainer {rank} is distributed, backend: {dist.get_backend()}"
    #         )
    #         self._policy.distributed()

    def get_checkpoint(self, *args, **kwargs):
        checkpoint = self.policy.get_checkpoint()
        checkpoint.update({"optimizer": self.optimizer.state_dict()})
        checkpoint.update({"steps": self._steps})
        # should include version steps
        return checkpoint

    def load_checkpoint(self, checkpoint, **kwargs):
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.policy.load_checkpoint(checkpoint, **kwargs)
