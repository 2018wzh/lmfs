import numpy as np
import torch


def get_batch(
    dataset: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    offsets = starts[:, None] + np.arange(context_length + 1)[None, :]
    tokens = torch.from_numpy(np.asarray(dataset[offsets], dtype=np.int64)).to(device)
    return tokens[:, :-1], tokens[:, 1:]
