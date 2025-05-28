from dataclasses import dataclass, field
import torch

@dataclass
class Args:

    # Stim 
    error_rates: list[float] = field(default_factory=lambda: [0.001, 0.002, 0.003, 0.004, 0.005])
    t: list[int] = field(default_factory=lambda: [99])
    dt: int = 2
    distance: int = 5
    sliding: bool = True
    k: int = 20
    seed: int | None = None
    norm: float | int = torch.inf

    # Torch
    device: torch.device = field(
    default_factory=lambda: torch.device(
        
        "cuda" if torch.cuda.is_available() else
        "cpu"
    ))
    batch_size: int = 2048
    n_batches: int = 256
    n_epochs: int = 600
    lr: float = 1e-3
    min_lr: float = 1e-4

    # Model
    embedding_features: list = field(default_factory=lambda: [5, 32, 64, 128, 256])
    hidden_size: int = 128 
    n_gru_layers: int = 4
