import time
import numpy as np
import pymatching
import torch
from torch.utils.data import random_split

from args import Args
from dataloader_ibm import IBMSampler

class MWPMDecoder:
    """
    Decoder using Minimum Weight Perfect Matching for syndrome-based error correction.
    """
    def __init__(self, args: Args, weight_scheme: str = "uniform") -> None:
        self.args = args
        self.distance = args.distance
        self.t = args.t[0] - 1
        self.simulator_backend = args.simulator_backend
        self.validation_ratio = args.val_fraction

        self.weight_scheme = weight_scheme
        self.matcher = pymatching.Matching()
        self.sampler = IBMSampler(distance=self.distance,
                                  t=self.t + 1,
                                  simulator=self.simulator_backend)

    def _load_job_data(self) -> None:
        """
        Load syndromes and flips from IBM sampler and compute mask for trivial syndromes.
        """
        t0 = time.perf_counter()
        self.syndromes, self.flips = self.sampler.load_jobdata()
        # True where any detection event occurred
        self.nontrivial_mask = np.any(self.syndromes, axis=1)
        self.total_shots = self.syndromes.shape[0]
        trivial_share = np.mean(~self.nontrivial_mask)
        print(f"Loaded data '{self.sampler.filename}' (d={self.distance}, t={self.t}) "
              f"with {self.total_shots} shots ({trivial_share*100:.1f}% trivial) in {time.perf_counter() - t0:.2f}s.")
        
    def _get_edges(self):

        # if weight_scheme == 'p_ij':
        #     with open(self.run+'_data/Error_matrix/error_matrix_t_'+self.backend_name+'_'+str(self.code_distance)
        #                     +'_'+str(self.shots)+'_'+str(self.t)+'_'+self.version+'.json' , 'r') as infile:
        #         error_matrix_t = np.array(json.load(infile))
        #         error_matrix_t[error_matrix_t == 0] = 0.0000001
        #         error_matrix_t[error_matrix_t < 0] = 0.0000001
        #         weight_matrix = -np.log(error_matrix_t)
        if self.weight_scheme == 'uniform':
            weight_matrix = np.ones(((self.t+1)*(self.distance-1),(self.t+1)*(self.distance-1)))

    #Add space-like edges:
        for i in range(0,(self.t+1)*(self.distance-1), self.distance-1):
            self.matcher.add_boundary_edge(i, weight=weight_matrix[i][i+1], fault_ids={0} ,merge_strategy='replace')
            for j in range(self.distance-2):
                self.matcher.add_edge(i+j,i+j+1, weight=weight_matrix[i+j][i+j+1], fault_ids={j+1}, merge_strategy='replace')
            self.matcher.add_boundary_edge(i+j+1, weight=weight_matrix[i+j][i+j+1], fault_ids={self.distance-1}, merge_strategy='replace')

    #Add nearest neighbour time-like edges:
        for i in range(0,(self.t+1) * (self.distance-1) - self.distance+1, self.distance-1):
            for j in range(self.distance-2):
                self.matcher.add_edge(i+j,i+j+self.distance-1, weight=weight_matrix[i+j][i+j+self.distance-1], merge_strategy='replace')
                self.matcher.add_edge(i+j+1,i+j+1+self.distance-1, weight=weight_matrix[i+j+1][i+j+1+self.distance-1], merge_strategy='replace')


    def _evaluate_predictions(self) -> float:
        """
        Decode validation set and compute logical accuracy, including trivial shots.

        Returns:
            logical_accuracy (float): Accuracy over the full validation set.
        """
        total_samples = self.syndromes.shape[0]
        val_count = int(self.validation_ratio * total_samples)
        train_count = total_samples - val_count

        # Random split with fixed seed
        train_set, val_set = random_split(
            self.syndromes,
            [train_count, val_count],
            generator=torch.Generator().manual_seed(42)
        )

        val_syndromes = self.syndromes[val_set.indices]
        val_flips = self.flips[val_set.indices]

        # Filter out trivial syndromes
        nontrivial = np.any(val_syndromes, axis=1)
        syndromes_nt = val_syndromes[nontrivial]
        flips_nt = val_flips[nontrivial]

        # Batch decode: returns pairs (bit_prediction, _) per syndrome
        predictions = self.matcher.decode_batch(syndromes_nt)

        # Compare only the logical flip at final time step
        actual = flips_nt[:, -1]
        predicted = predictions[:, 0]
        correct = np.sum(actual == predicted)
        trivial_count = np.sum(~nontrivial)

        pred_acc = correct / len(syndromes_nt) if len(syndromes_nt) > 0 else 0.0
        print(f"Nontrivial prediction accuracy: {pred_acc:.3f}")

        logical_accuracy = (correct + trivial_count) / val_count
        print(f"Overall logical accuracy: {logical_accuracy:.3f}")
        return logical_accuracy

    def decode(self) -> float:
        """
        Full decoding pipeline: load data, build graph, predict, and report accuracy.
        """
        self._load_job_data()
        self._get_edges()
        return self._evaluate_predictions()


if __name__ == "__main__":
    args = Args(t=[6], distance=3, sliding=True, dt=2, simulator_backend=False)
    decoder = MWPMDecoder(args)
    accuracy = decoder.decode()
    print(f"Decoder completed with logical accuracy: {accuracy:.3f}")
