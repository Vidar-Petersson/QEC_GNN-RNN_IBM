import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from decoder_binary.gru_decoder import GRUDecoder
from args import Args
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=15)
    parser.add_argument('--t', type=int, default=50)
    parser.add_argument('--dt', type=int, default=2)
    args_cli = parser.parse_args()

    args = Args(
        distance=args_cli.d,
        t=args_cli.t,
        load_distance=None,
        noise_angle=0.3307,
        dt=args_cli.dt,
        sliding=True,
        batch_size=512,
        embedding_features=[2, 32, 64, 128, 256], # Change first element to 3 to include IQ-data
        hidden_size=128,
        n_gru_layers=4,
        seed=42, 
        simulator_backend = False,
        val_fraction=0.1,
        sub_dir="/turning_the_knob",
    )
    
    model_name = "train_final_t_d15_t50_dt2_alpha0.1653_250813_133742.pt"
    decoder = GRUDecoder(args)
    model = torch.load(f"./models/{model_name}", weights_only=True, map_location=args.device)
    decoder.load_state_dict(model['model_state_dict'])

    decoder.to(args.device)  # Move model to MPS or appropriate device
    avg_loss, physical_acc, logical_acc = decoder.test_model()