from collections.abc import Iterable

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logits = shifted.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()


@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> torch.Tensor:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total_norm = torch.sqrt(torch.stack([g.float().pow(2).sum() for g in grads]).sum())
    scale = torch.clamp(max_l2_norm / (total_norm + eps), max=1.0)
    for g in grads:
        g.mul_(scale)
    return total_norm
