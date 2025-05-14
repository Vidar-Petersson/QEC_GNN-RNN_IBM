import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from gru_decoder import GRUDecoder
from data import Args
from utils import TrainingLogger
import torch

if __name__ == "__main__":
    # Let's train a distance 5 model:
    args = Args(
        distance=5,
        error_rates=[0.001, 0.002, 0.003, 0.004, 0.005],
        t=[49],
        dt=2,
        sliding=True,
        batch_size=64,
        n_batches=10,
        n_epochs=10,
        embedding_features=[5, 32, 64, 128, 256],
        hidden_size=128,
        n_layers=4,
    )

    logger = TrainingLogger(logfile="logs", statsfile="stats")
    decoder_d5 = torch.compile(GRUDecoder(args))
    decoder_d5.train_model(logger, save="decoder_d5")
