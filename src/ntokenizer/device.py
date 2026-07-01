"""Shared device auto-detection."""

import torch


def auto_device(preferred: str = "") -> torch.device:
    """Return `preferred` as a torch.device if given, else auto-detect mps > cuda > cpu."""
    if preferred:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
