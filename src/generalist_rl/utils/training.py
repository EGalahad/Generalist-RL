from generalist_rl.api.datatypes import SampleBatch
import torch
from generalist_rl.utils.namedarray import recursive_apply


def minibatch_generator(samples: SampleBatch, num_epochs: int, num_minibatch: int, shuffle: bool = True):
    # apply flatten to all tensors
    samples = recursive_apply(samples, lambda x: x.view(-1, x.shape[-1]))
    device = samples.obs['obs'].device
    batch_size = samples.obs['obs'].shape[0]
    indices = torch.randperm(batch_size, requires_grad=False, device=device) if shuffle else torch.arange(batch_size, requires_grad=False, device=device)
    minibatch_size = batch_size // num_minibatch
    assert batch_size % num_minibatch == 0, f"batch_size {batch_size} must be divisible by num_minibatch {num_minibatch}"
    for _ in range(num_epochs):
        for i in range(num_minibatch):
            start_idx = i * minibatch_size
            end_idx = (i + 1) * minibatch_size
            minibatch_indices = indices[start_idx:end_idx]
            yield samples[minibatch_indices]