import os
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch_geometric.nn import GraphConv
import torch
import logging
import time
from typing import Dict
StateDict = Dict[str, torch.Tensor]

def group(x, label_map):
        """
        Groups graphs according to which batch element they belong to. 

        Args:
        x: tensor of shape [n_graphs, embedding size]. 
        label_map: tensor of shape [n_graphs]. 
    
        Returns: 
        A tensor of shape [batch size, g, embedding size] where
            g represents the number of graphs belonging to a batch element. 
            If t = 24 and dt = 5, then g = 5, i.e. g = (t + 1) / dt.
            Batch elements may contain less than (t + 1) / dt graphs. 
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
        logging.basicConfig(filename=f"./logs/{logfile}", level=logging.INFO, format="%(message)s")
        self.logs = []
        self.statsfile = statsfile
        self.best_accuracy = 0 
    
    def on_epoch_begin(self, epoch):
        self.t0 = time.perf_counter()
        self.epoch = epoch
        logging.info(f"EPOCH {epoch} starting")
    
    def on_epoch_end(self, logs=None):
        epoch_time = time.perf_counter() - self.t0
        if logs["accuracy"] > self.best_accuracy:
            self.best_accuracy = logs["accuracy"]
        logging.info(
            f"EPOCH {self.epoch} finished in {epoch_time:.3f} seconds with lr = {logs['lr']:.2e}:\n"
            f"\tloss = {logs['loss']:.5f}, accuracy = {logs['accuracy']:.4f} ({self.best_accuracy:.4f})\n"
            f"\tclass 0: mean = {logs['zero_mean']:.4f} std = {logs['zero_std']:.4f} "
            f"fraction = {logs['noflip']:.4f}\n"
            f"\tclass 1: mean = {logs['one_mean']:.4f} std = {logs['one_std']:.4f} "
            f"fraction = {1 - logs['noflip']:.4f}\n"
            f"\tmodel time = {logs['model_time']:.2f} seconds, "
            f"data time = {logs['data_time']:.2f} seconds"
        )
        self.logs.append(logs)

    def on_training_begin(self, args):
        logging.info(f"Training with t = {args.t}, dt = {args.dt}, distance = {args.distance}")
    
    def on_training_end(self):
        stats = np.vstack((
            [logs["model_time"] for logs in self.logs],
            [logs["data_time"] for logs in self.logs],
            [logs["lr"] for logs in self.logs],
            [logs["loss"] for logs in self.logs],
            [logs["accuracy"] for logs in self.logs],
            [logs["zero_mean"] for logs in self.logs],
            [logs["zero_std"] for logs in self.logs],
            [logs["one_mean"] for logs in self.logs],
            [logs["one_std"] for logs in self.logs],
            [logs["noflip"] for logs in self.logs],
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
