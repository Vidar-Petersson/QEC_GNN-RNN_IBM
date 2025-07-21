from dataclasses import dataclass, field
import torch

@dataclass
class Args:

    # Repetition code data 
    error_rates: list[float] = field(default_factory=lambda: [0.001, 0.002, 0.003, 0.004, 0.005]) # Only applicable for stim
    t: list[int] = field(default_factory=lambda: [99]) # Perhaps change from list to int datatype
    dt: int = 2
    distance: int = 3
    load_distance: None | int = None
    sliding: bool = True
    k: int = 20
    seed: int | None = None
    norm: float | int = torch.inf
    train_all_times: bool = True # Evaluate if working correctly
    pretrained_checkpoint: str = None   # Sökväg till .pt-fil att förträna från
    resume: bool = False               # Om True, läs in optimizer‑ och scheduler‑status
    simulator_backend: bool = True
    patience: int = 20

    # Torch
    device: torch.device = field(
    default_factory=lambda: torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    ))
    batch_size: int = 2048
    n_batches: int = 256 # Should be dynamic and depend on the number of shots in job
    n_epochs: int = 600
    lr: float = 1e-3
    min_lr: float = 1e-4

    # Model
    embedding_features: list = field(default_factory=lambda: [2, 32, 64, 128, 256])
    hidden_size: int = 128 
    n_gru_layers: int = 4

    # Training
    val_fraction: float = 0.1 # Portion of data used for validation
    pre_train: bool = False # If simulated data should be used for pre-training network
    log_wandb: bool = False
