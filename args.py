from dataclasses import dataclass, field
import torch
from typing import List, Optional, Union

@dataclass
class Args:
    """Configuration parameters for repetition code data, graph creation, and model training."""

    # Repetition code data parameters
    error_rates: List[float] = field(default_factory=lambda: [0.001, 0.002, 0.003, 0.004, 0.005])  # Physical error rates (STIM only, not used for Aer simulator)
    t: int = 6  # Number of final detectors, including those inferred from perfect syndromes
    dt: int = 2  # Temporal step between detection layers
    distance: int = 3  # Code distance of the repetition code
    noise_angle: float = 0.0  # Noise rotation angle (in radians)
    sub_dir: Optional[str] = None  # Optional subdirectory for saving/loading data
    simulator_backend: bool = True  # True = simulator backend, False = hardware data

    # Detection event extraction
    load_distance: Optional[int] = None  # If set, overrides `distance` when matching jobdata filename and subsamples to distance
    detection_threshold: float = 0.5  # Threshold for binary classification of detection events

    # Graph creation parameters
    sliding: bool = True  # Use sliding time-window graph construction
    k: int = 20  # Number of nearest neighbors for edge creation
    seed: int = 42  # Random seed for reproducibility
    norm: Union[float, int] = torch.inf  # Norm used for nearest-neighbor calculations

    # Training parameters
    train_all_times: bool = False  # Train using all time steps, not sure if working properly
    pretrained_checkpoint: Optional[str] = None  # Path to `.pt` file with pretrained weights
    resume: bool = False  # Resume optimizer/scheduler state from checkpoint, do not use if utilizing transfer learning or pre training on simulated data
    patience: int = 20  # Epochs without validation improvement before early stopping
    val_fraction: float = 0.1  # Fraction of total data used for validation, should always be the same permutation due to seed
    log_wandb: bool = False  # Log training metrics to Weights & Biases

    # Torch-specific parameters
    device: torch.device = field(  # Torch device for computation
        default_factory=lambda: torch.device(
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
    )
    batch_size: int = 512  # Samples per training batch
    n_batches: int = 256  # Batches per epoch (should be dynamic based on job shots), currently not used
    n_epochs: int = 600  # Max training epochs (unless early stopping)
    lr: float = 1e-3  # Initial learning rate
    min_lr: float = 1e-4  # Minimum learning rate during decay

    # Model architecture parameters
    embedding_features: List[int] = field(default_factory=lambda: [2, 32, 64, 128, 256])  # Features per embedding layer, change first element to 3 for IQ-data
    hidden_size: int = 128  # Hidden state size for GRU layers
    n_gru_layers: int = 4  # Number of stacked GRU layers
