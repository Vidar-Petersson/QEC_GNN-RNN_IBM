import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch_geometric.nn import GraphConv

from threading import Thread
from queue import Queue
from typing import Dict, Optional, Tuple, Callable

StateDict = Dict[str, torch.Tensor]

def save_checkpoint(model: nn.Module,
                    path: str,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    epoch: int) -> None:
    """
    Save a model checkpoint containing model, optimizer, and scheduler states.

    Args:
        model: The PyTorch model to save.
        path: Path to save checkpoint (.pt or .pth file).
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler instance.
        epoch: Current epoch number.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path)

def load_checkpoint(model: nn.Module,
                    path: str,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                    resume: bool = False,
                    device: str = "cpu") -> int:
    """
    Load a model checkpoint from disk.

    Args:
        model: Model instance to load weights into.
        path: Path to checkpoint file.
        optimizer: (Optional) Optimizer instance for resuming training.
        scheduler: (Optional) Scheduler instance for resuming training.
        resume: If True, also load optimizer and scheduler states.
        device: Device to map checkpoint to ("cpu" or "cuda").

    Returns:
        The epoch number stored in the checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if resume and optimizer and scheduler:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"]

def generate_batches_async(graph_creator, mode: str, max_prefetch: int = 5) -> Tuple[Thread, Callable[[], Optional[dict]]]:
    """
    Asynchronously prefetch batches from `graph_creator.generate_batches(mode)` into a queue.

    Args:
        graph_creator: Object with a `.generate_batches(mode)` method yielding batches.
        mode: Mode string to pass to `.generate_batches`.
        max_prefetch: Max number of batches to hold in the queue.

    Returns:
        thread: Background thread running the producer.
        get_next_batch: Callable returning the next batch or None when done.
    """
    queue = Queue(max_prefetch)

    def _producer():
        for batch in graph_creator.generate_batches(mode):
            queue.put(batch)
        queue.put(None)  # Sentinel to signal completion

    thread = Thread(target=_producer, daemon=True)
    thread.start()

    def get_next_batch():
        return queue.get()

    return thread, get_next_batch

def group(x, label_map):
        """
        Groups graphs according to which batch element they belong to. 

        Args:
        x: tensor of shape [n_graphs, embedding size]. 
        label_map: tensor of shape [n_graphs]. 
    
        Returns: 
        A tensor of shape [batch size, g, embedding size] where
            g represents the number of graphs belonging to a batch element. 
            If t = 24 and dt = 5, then g = 5, i.e. g = t - dt + 2.
            Batch elements may contain less than t - dt + 2 graphs. 
            This happens when there are no detection events in a chunk. 
            For instance, if t = 24 and dt = 5, and no detection
            events occur between timesteps 0 and 4, there would
            be no graph for this chunk. Therefore, any "missing" graphs are 
            replaced with zeros, such that the dimensions work out properly. 
            The zero padding happens at the end of the sequence, e.g. if 
            g = 5 and some batch element consists only of graphs 2 and 3,
            the result would look like [2, 3, 0, 0, 0], where 2 and 3 
            represent the graph embeddings for graphs 2 and 3, and the zeros
            represent zero-padding.  
        """     
        counts = torch.unique(label_map[:, 0], return_counts=True)[-1]
        grouped = torch.split(x, list(counts))
        padded = pad_sequence(grouped, batch_first=True)
        # padded has shape [batch, t, embedding_features[-1]]
        return pack_padded_sequence(padded, counts.cpu(), batch_first=True, enforce_sorted=False)

class GraphConvLayer(nn.Module):
    """
    Wrapper for a GraphConv layer with activation.
    """
    def __init__(self, in_features: int, out_features: int, act: nn.Module = nn.ReLU()):
        super().__init__()
        self.layer = GraphConv(in_features, out_features)
        self.act = act

    def forward(self, x, edge_index, edge_attr):
        x = self.layer(x, edge_index, edge_attr)
        return self.act(x)

class TrainingLogger:
    """
    Logs training progress to file and stores statistics.
    """
    def __init__(self, logfile: Optional[str] = None, statsfile: Optional[str] = None):
        if logfile:
            os.makedirs("./logs", exist_ok=True)
            logging.basicConfig(
                filename=f"./logs/{logfile}.out",
                level=logging.INFO,
                format="%(message)s"
            )

        self.logs = []
        self.statsfile = statsfile
        self.best_accuracy = 0

    def on_training_begin(self, args):
        logging.info(f"Training with t={args.t}, dt={args.dt}, distance={args.distance}")

    def on_epoch_begin(self, epoch: int):
        self.t0 = time.perf_counter()
        self.epoch = epoch
        logging.info(f"EPOCH {epoch} starting")

    def on_epoch_end(self, logs: Dict[str, float]):
        epoch_time = time.perf_counter() - self.t0

        val_acc = logs["val_acc"]
        train_acc = logs["train_acc"]

        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc

        logging.info(
            f"EPOCH {self.epoch} finished in {epoch_time:.3f}s, "
            f"LR={logs['learning_rate']:.2e}:\n"
            f"\tTrain loss={logs['train_loss']:.5f}, acc={train_acc:.4f}\n"
            f"\tVal loss={logs['val_loss']:.5f}, acc={val_acc:.4f} (best={self.best_accuracy:.4f})\n"
            f"\tModel time={logs.get('model_time', 0):.2f}s, "
            f"Data time={logs.get('data_time', 0):.2f}s"
        )
        self.logs.append(logs)

    def on_training_end(self):
        stats = np.vstack((
            [l.get("model_time", 0) for l in self.logs],
            [l.get("data_time", 0) for l in self.logs],
            [l["learning_rate"] for l in self.logs],
            [l["train_loss"] for l in self.logs],
            [l["train_acc"] for l in self.logs],
            [l["val_loss"] for l in self.logs],
            [l["val_acc"] for l in self.logs],
        ))

        if self.statsfile:
            os.makedirs("./stats", exist_ok=True)
            np.save(f"./stats/{self.statsfile}", stats)

def plot_model_confidence(final_preds: torch.Tensor,
                          last_labels: torch.Tensor,
                          num_bins: int = 30) -> None:
    """
    Plot histogram of model prediction confidence and failure rate per bin.

    Args:
        final_preds: Predicted probabilities for class 1, shape (N,).
        last_labels: Ground truth labels (0 or 1), shape (N,).
        num_bins: Number of histogram bins.
    """
    pred_labels = (final_preds >= 0.5).int()
    correct_mask = (pred_labels == last_labels)

    # Confidence for predicted class
    confidence = torch.where(correct_mask, final_preds, 1 - final_preds).clamp(1e-20, 1 - 1e-20)
    confidence_np = confidence.cpu().numpy()
    correct_np = correct_mask.cpu().numpy()

    counts, bin_edges = np.histogram(confidence_np, bins=num_bins)
    bin_indices = np.digitize(confidence_np, bins=bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Failure rate per bin
    failure_rate_per_bin = [
        (1 - np.mean(correct_np[bin_indices == b])) if np.any(bin_indices == b) else np.nan
        for b in range(num_bins)
    ]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(confidence_np[correct_np], bins=bin_edges, alpha=0.7, label="Correct predictions")
    ax1.hist(confidence_np[~correct_np], bins=bin_edges, alpha=0.7, label="Incorrect predictions")
    ax1.set_xlabel("Predicted probability of logical flip")
    ax1.set_ylabel("Frequency")
    ax1.set_yscale("log")
    ax1.set_title("Model confidence and failure rate per bin")
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Secondary axis for failure rate
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, failure_rate_per_bin, color='red', marker='o', label='Failure rate per bin')
    ax2.set_ylabel("Failure rate")
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')

    plt.show()

def standard_deviation(p, n):
    """
    Standard deviation of the Binomial distribution.
    https://en.wikipedia.org/wiki/Binomial_distribution
    """
    return np.sqrt(p * (1 - p) / n)