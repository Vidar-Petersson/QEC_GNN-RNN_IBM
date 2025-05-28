import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from gru_decoder import GRUDecoder
from data import Args
from utils import TrainingLogger
import torch
from datetime import datetime

if __name__ == "__main__":
    d = 5
    t = 49
    dt = 2
    args = Args(
        distance=d,
        error_rates=[0.001, 0.002, 0.003, 0.004, 0.005],
        t=[t],
        dt=dt,
        sliding=True,
        batch_size=64,
        n_batches=100,
        n_epochs=100,
        embedding_features=[5, 32, 64, 128, 256],
        hidden_size=128,
        n_gru_layers=4,
    )
    current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")
    model_name = 'd' + str(d) + '_t' + str(t) + '_dt' + str(dt) + '_' + current_datetime

    logger = TrainingLogger(logfile=model_name, statsfile=model_name)
    decoder = GRUDecoder(args)
    decoder.to(args.device)  # Move model to MPS or appropriate device
    decoder = torch.compile(decoder)  # Then compile
    decoder.train_model(logger, save=model_name)
