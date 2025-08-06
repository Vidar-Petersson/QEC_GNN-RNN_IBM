import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from gru_decoder_IQ import GRUDecoder
from args import Args
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=15)
    parser.add_argument('--t', type=int, default=50)
    parser.add_argument('--dt', type=int, default=2)
    parser.add_argument('--p', type=float, default=0.001)
    parser.add_argument('--n_iter', type=int, default=1) # iterationer har ingen betydelse eftersom all data redan är genererad
    args_cli = parser.parse_args()

    args = Args(
        distance=args_cli.d,
        t=[args_cli.t],
        load_distance=49,
        noise_angle=0.0251,
        dt=args_cli.dt,
        sliding=True,
        batch_size=2048,
        embedding_features=[2, 32, 64, 128, 256],
        hidden_size=128,
        n_gru_layers=4,
        seed=42, 
        simulator_backend = False,
        val_fraction=0.1,
    )
    
    model_name = "train_final_t_d15_t50_dt2_alpha0.5969_250725_230444.pt"
    decoder = GRUDecoder(args)
    model = torch.load(f"./models/{model_name}", weights_only=True, map_location=args.device)
    decoder.load_state_dict(model['model_state_dict'])

    decoder.to(args.device)  # Move model to MPS or appropriate device
    avg_loss, physical_acc, logical_acc = decoder.test_model()