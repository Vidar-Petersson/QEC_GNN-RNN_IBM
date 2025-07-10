import os
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch_geometric.nn import GraphConv
import torch
import logging
import time
from typing import Dict
from threading import Thread
from queue import Queue
StateDict = Dict[str, torch.Tensor]

def save_checkpoint(self, path: str, optimizer, scheduler, epoch: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': self.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)

def load_checkpoint(self, path: str, optimizer=None, scheduler=None, resume: bool = False):
    checkpoint = torch.load(path, map_location=self.args.device)
    self.load_state_dict(checkpoint['model_state_dict'])
    if resume and optimizer and scheduler:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']

def generate_batches_async(gc, mode):
    """
    Asynchronously generates batches by calling gc.generate_batches(mode) in a background thread.

    Returns:
        - thread: The thread object performing the batch generation.
        - get_batches: A function that blocks until the batches are ready and then returns them.
    """
    q = Queue()

    def _target():
        result = gc.generate_batches(mode=mode)
        q.put(result)

    thread = Thread(target=_target)
    thread.start()
    
    return thread, lambda: q.get()

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
    def __init__(self, in_features, out_features, act=nn.ReLU()):
        super().__init__()
        self.layer = GraphConv(in_features, out_features)
        self.act = act
    
    def forward(self, x, edge_index, edge_attr):
        x = self.layer(x, edge_index, edge_attr)
        return self.act(x)

class TrainingLogger:
    def __init__(self, logfile=None, statsfile=None):
        if logfile:
            os.makedirs("./logs", exist_ok=True)
        logging.basicConfig(filename=f"./logs/{logfile}.out", level=logging.INFO, format="%(message)s")
        self.logs = []
        self.statsfile = statsfile
        self.best_accuracy = 0 
    
    def on_training_begin(self, args):
        logging.info(f"Training with t = {args.t}, dt = {args.dt}, distance = {args.distance}")
    
    def on_epoch_begin(self, epoch):
        self.t0 = time.perf_counter()
        self.epoch = epoch
        logging.info(f"EPOCH {epoch} starting")
    
    def on_epoch_end(self, logs=None):
        epoch_time = time.perf_counter() - self.t0

        val_acc = logs["val_acc"]
        val_loss = logs["val_loss"]
        train_acc = logs["train_acc"]
        train_loss = logs["train_loss"]

        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc

        logging.info(
            f"EPOCH {self.epoch} finished in {epoch_time:.3f} seconds with learning_rate = {logs['learning_rate']:.2e}:\n"
            f"\tTrain   loss = {train_loss:.5f}, accuracy = {train_acc:.4f}\n"
            f"\tVal     loss = {val_loss:.5f}, accuracy = {val_acc:.4f} (best = {self.best_accuracy:.4f})\n"
            f"\tModel time = {logs.get('model_time', 0):.2f} seconds, "
            f"Data time = {logs.get('data_time', 0):.2f} seconds"
        )
        self.logs.append(logs)

    def on_training_end(self):
        stats = np.vstack((
            [logs.get("model_time", 0) for logs in self.logs],
            [logs.get("data_time", 0) for logs in self.logs],
            [logs["learning_rate"] for logs in self.logs],
            [logs["train_loss"] for logs in self.logs],
            [logs["train_acc"] for logs in self.logs],
            [logs["val_loss"] for logs in self.logs],
            [logs["val_acc"] for logs in self.logs],
        ))
        if self.statsfile:
            os.makedirs("./stats", exist_ok=True)
            np.save(f"./stats/{self.statsfile}", stats)

def standard_deviation(p, n):
    """
    Standard deviation of the Binomial distribution.
    https://en.wikipedia.org/wiki/Binomial_distribution
    """
    return np.sqrt(p * (1 - p) / n)
