import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from gru_decoder import GRUDecoder
from data import Dataset, Args
from mwmp import test_mwpm
import torch
import argparse
# python examples/test_nn.py --d 5 --t 49 --dt 2 --p 0.001 --n_iter 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--t', type=int, default=49)
    parser.add_argument('--dt', type=int, default=2)
    parser.add_argument('--p', type=float, default=0.001)
    parser.add_argument('--n_iter', type=int, default=100)
    args_cli = parser.parse_args()

    # Let's load and test the distance 5 model:
    args = Args(
        distance=args_cli.d,
        error_rates=[args_cli.p],
        t=[args_cli.t],
        dt=args_cli.dt,
        sliding=True,
        batch_size=128,
        embedding_features=[5, 32, 64, 128, 256],
        hidden_size=128,
        n_gru_layers=4,
        seed=42 
    )

    decoder = GRUDecoder(args)
    decoder.load_state_dict(torch.load("./models/d5_t49_dt2_250528_142527.pt", weights_only=True))
    n_iter = args_cli.n_iter
    decoder.to(args.device)  # Move model to MPS or appropriate device
    decoder.test_model(Dataset(args), n_iter=n_iter)
    test_mwpm(Dataset(args), n_iter=n_iter)
