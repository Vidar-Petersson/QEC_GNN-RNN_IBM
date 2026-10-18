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
    # By default, the model trained at alpha=0.1653 is evaluated on the alpha=0.3307 dataset (cross-noise test)
    parser.add_argument('--alpha', type=float, default=0.3307, help="Noise angle of the dataset to evaluate on")
    parser.add_argument('--model', type=str, default="train_final_t_d15_t50_dt2_alpha0.1653_250813_133742.pt",
                        help="Checkpoint file in ./models")
    args_cli = parser.parse_args()

    args = Args(
        distance=args_cli.d,
        t=args_cli.t,
        load_distance=None,
        noise_angle=args_cli.alpha,
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

    decoder = GRUDecoder(args)
    model = torch.load(f"./models/{args_cli.model}", weights_only=True, map_location=args.device)
    decoder.load_state_dict(model['model_state_dict'])

    decoder.to(args.device)  # Move model to MPS or appropriate device
    avg_loss, physical_acc, logical_acc = decoder.test_model()
