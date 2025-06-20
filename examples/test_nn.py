import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from gru_decoder import GRUDecoder
from data_ibm import Dataset, Args
from mwmp import test_mwpm
import torch
import argparse
# python examples/test_nn.py --d 5 --t 49 --dt 2 --p 0.001 --n_iter 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=3)
    parser.add_argument('--t', type=int, default=49)
    parser.add_argument('--dt', type=int, default=2)
    parser.add_argument('--p', type=float, default=0.001)
    parser.add_argument('--n_iter', type=int, default=1) # iterationer har ingen betydelse eftersom all data redan är genererad
    args_cli = parser.parse_args()

    args = Args(
        distance=args_cli.d,
        error_rates=[args_cli.p],
        t=[args_cli.t],
        dt=args_cli.dt,
        sliding=True,
        batch_size=100,
        embedding_features=[5, 32, 64, 128, 256],
        hidden_size=128,
        n_gru_layers=4,
        seed=42 
    )

    decoder = GRUDecoder(args)
    decoder.load_state_dict(torch.load("./models/train_final_t_d3_t6_dt2_250612_162334.pt", weights_only=True, map_location=args.device))
    n_iter = args_cli.n_iter
    decoder.to(args.device)  # Move model to MPS or appropriate device
    accuracies = []
    for t in [14]:#, 14, 24]:#, 49, 74, 99, 249, 499, 749, 999]:
        print('Starting with t=',t)
        args = Args(
            distance=args_cli.d,
            error_rates=[args_cli.p],
            t=[t],
            dt=args_cli.dt,
            sliding=True,
            batch_size=200000,
            embedding_features=[5, 32, 64, 128, 256],
            hidden_size=128,
            n_gru_layers=4,
            seed=42 
        )
        acc, std = decoder.test_model(Dataset(args), n_iter=n_iter)
        accuracies.append(acc)
        print(t, acc)
    print(accuracies)