"""Default device/dtype for GP tensors, so surrogate work can run on a GPU.

Optimizers build their GP inputs with ``tensor_kw()``; one :func:`use_device` call retargets them
all. Default is CPU float64; on CUDA the dtype switches to float32 unless overridden (consumer
GPUs throttle float64), and the GP's escalating-jitter Cholesky keeps float32 stable at typical
design sizes.
"""

import torch

_DEVICE = torch.device("cpu")
_DTYPE = torch.float64


def use_device(device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> None:
    """Set the global device (and optionally dtype) for GP tensors."""
    global _DEVICE, _DTYPE
    if device is not None:
        _DEVICE = torch.device(device)
        _DTYPE = dtype or (torch.float32 if _DEVICE.type == "cuda" else torch.float64)
    elif dtype is not None:
        _DTYPE = dtype


def tensor_kw() -> dict:
    """The ``{device, dtype}`` kwargs for ``torch.tensor`` on GP inputs."""
    return {"device": _DEVICE, "dtype": _DTYPE}


def device() -> torch.device:
    return _DEVICE
