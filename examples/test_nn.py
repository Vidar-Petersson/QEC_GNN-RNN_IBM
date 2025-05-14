import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from gru_decoder import GRUDecoder
from data import Dataset, Args
from mwmp import test_mwpm
import torch

if __name__ == "__main__":
    # Let's load and test the distance 5 model:
    args = Args(
        distance=5,
        error_rates=[0.001],
        t=[99],
        dt=2,
        sliding=True,
        batch_size=128,
        embedding_features=[5, 32, 64, 128, 256],
        hidden_size=128,
        n_layers=4,
        seed=42 
    )

    decoder_d5 = GRUDecoder(args)
    decoder_d5.load_state_dict(torch.load("./models/distance5.pt", weights_only=True))
    
    n_iter = 100
    decoder_d5.test_model(Dataset(args), n_iter=n_iter)
    test_mwpm(Dataset(args), n_iter=n_iter)
