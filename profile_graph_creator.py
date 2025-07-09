import cProfile
import pstats
from args import Args
from graph_creator import GraphCreator

def main():
    args = Args(error_rates=[0.002], t=[6], distance=3, sliding=True, dt=2, simulator_backend=False)
    gc = GraphCreator(args)
    gc.train_val_split(seed=42)

    # Profilera generate_batches (training)
    gc.profile_batches(mode="training")

if __name__ == "__main__":
    cProfile.run("main()", filename="profile_stats.prof")
