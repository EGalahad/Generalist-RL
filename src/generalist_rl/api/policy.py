from typing import Dict, List, Union
from generalist_rl.api.datatypes import PolicyState, AnalyzedResult, RolloutRequest, RolloutResult, SampleBatch

import torch
import torch.nn as nn

class Policy:
    @property
    def default_policy_state(self) -> PolicyState:
        """Default value of policy state."""
        raise NotImplementedError()

    @property
    def version(self) -> int:
        """Current version of the policy."""
        raise NotImplementedError()

    @property
    def net(self) -> Union[nn.Module, List[nn.Module]]:
        """Neural Network of the policy."""
        raise NotImplementedError()
    
    @property
    def device(self) -> torch.device:
        """Device of the policy."""
        raise NotImplementedError()

    def analyze(self, sample: SampleBatch, **kwargs) -> AnalyzedResult:
        """Generate outputs required for loss computation during training,
            e.g. value target and action distribution entropies.
        Args:
            sample (namedarraytuple): Customized namedarraytuple containing
                all data required for loss computation.

        Returns:
            training_seg (namedarraytuple): Data generated for loss computation.
        """
        raise NotImplementedError()

    def reanalyze(self, sample: SampleBatch, **kwargs) -> SampleBatch:
        """Reanalyze the sample with the current parameters.
        Args:
            sample (namedarraytuple): sample to be reanalyzed.

        Returns:
            Reanalyzed sample (algorithm.trainer.SampleBatch).
        """
        raise NotImplementedError()

    def rollout(self, requests: RolloutRequest, **kwargs) -> RolloutResult:
        """Generate actions (and rnn hidden states) during evaluation.
        Args:
            requests: All request received from actor generated by env.step.
        Returns:
            RolloutResult: Rollout results to be distributed (namedarray).
        """
        raise NotImplementedError()

    def parameters(self):
        raise NotImplementedError()

    def get_checkpoint(self):
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint):
        raise NotImplementedError()

    def train_mode(self):
        raise NotImplementedError()

    def eval_mode(self):
        raise NotImplementedError()

    def inc_version(self):
        """Increase the policy version."""
        raise NotImplementedError()

    def distributed(self):
        """Make the policy distributed."""
        raise NotImplementedError()


class SingleModelPytorchPolicy(Policy):
    def __init__(self, neural_network: nn.Module):
        """Initialization method of SingleModelPytorchPolicy
        Args:
            neural_network: nn.module.

        Note:
            After initialization, access the neural network from property policy.net
        """
        # set device
        ## ray will set the environment variable CUDA_VISIBLE_DEVICES for each worker.
        ## and pytorch will use that environment variable to decide which GPU to use.
        # logger.debug(
        #     f"Environment variable CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}"
        # )
        # logger.debug(f"Torch cuda device count: {torch.cuda.device_count()}")
        # self._device = torch.device("cpu")
        # if torch.cuda.is_available():
        #     self._device = torch.cuda.current_device()
        # logger.info(f"Policy will use device: {self.device}")

        self._net: torch.nn.Module = neural_network.to(self.device)
        self._version = -1
    
    # def distributed(self, **kwargs):
    #     """Wrapper of Pytorch DDP method.
    #     Ref: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    #     """
    #     from torch.nn.parallel import DistributedDataParallel as DDP

    #     # import DDP globally will cause incompatible issue between CUDA and multiprocessing.
    #     if dist.is_initialized():
    #         if self.device == "cpu":
    #             self._net = DDP(self._net, **kwargs)
    #         else:
    #             # FIXME: @fuwei
    #             find_unused_parameters = self.__class__.__name__ == "AtariDQNPolicy"
    #             self._net = DDP(
    #                 self._net,
    #                 device_ids=[self.device],
    #                 output_device=self.device,
    #                 find_unused_parameters=find_unused_parameters,
    #                 **kwargs,
    #             )

    @property
    def version(self) -> int:
        """In single model policy, version tells the the number of trainer steps have been performed on the mode.
        Specially, -1 means the parameters are from arbitrary initialization.
        0 means the first version that is pushed by the trainer
        """
        return self._version

    @property
    def net(self) -> nn.Module:
        return self._net

    @property
    def device(self) -> torch.device:
        return self._device

    def inc_version(self):
        self._version += 1

    def parameters(self):
        return self._net.parameters(recurse=True)

    def train_mode(self):
        self._net.train()

    def eval_mode(self):
        self._net.eval()

    def load_checkpoint(self, checkpoint: Dict):
        """Load a checkpoint.
        If "steps" is missing in the checkpoint. We assume that the checkpoint is from a pretrained model. And
        set version to 0. So that the trainer side won't ignore the sample generated by this version.
        """
        self._version = checkpoint.get("steps", 0)
        self._net.load_state_dict(checkpoint["state_dict"])

    def get_checkpoint(self) -> Dict:
        # if dist.is_initialized():
        #     return {
        #         "steps": self._version,
        #         "state_dict": {
        #             k.replace("module.", ""): v.cpu()
        #             for k, v in self._net.state_dict().items()
        #         },
        #     }
        # else:
        #     return {
        #         "steps": self._version,
        #         "state_dict": {k: v.cpu() for k, v in self._net.state_dict().items()},
        #     }
            return {
                "steps": self._version,
                "state_dict": {k: v.cpu() for k, v in self._net.state_dict().items()},
            }