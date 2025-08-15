import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))  # Add parent dir for imports
import numpy as np
import pymatching

from args import Args
from decoder_iq.dataloader_ibm_IQ import IBMSampler
from training_utils import standard_deviation


class MWPMDecoder:
    """
    Minimum Weight Perfect Matching (MWPM) decoder for surface code syndrome data.

    This decoder uses PyMatching to construct a matching graph based on observed
    detection events from a quantum circuit, and predicts logical errors
    by decoding these using MWPM. Weights can be computed from pairwise detection correlations
    or assumed uniform.
    """

    def __init__(self, args: Args, weight_scheme: str = "uniform") -> None:
        """
        Initialize the MWPMDecoder.

        Parameters
        ----------
        args : Args
            Configuration object containing code parameters and backend settings.
        weight_scheme : str, optional
            Weight scheme for graph edges. Either 'uniform' or 'p_ij' for correlation-based weights.
        """
        self.args = args
        self.distance = args.distance
        self.t = args.t - 1  # Number of rounds minus one (for indexing)
        self.simulator_backend = args.simulator_backend
        self.validation_ratio = args.val_fraction
        self.load_distance = args.load_distance
        self.weight_scheme = weight_scheme

        self.matcher = pymatching.Matching()
        self._load_job_data()

    def _load_job_data(self) -> None:
        """
        Load syndrome and logical flip data from the sampler.
        Also computes a mask identifying trivial (all-zero) syndromes.
        """
        self.sampler = IBMSampler(self.args)
        self.detections, self.flips, _, _ = self.sampler.load_jobdata(verbose=True)

    def train_val_split(self) -> None:
        num_total = self.detections.shape[0]
        self.val_size = int(num_total * self.validation_ratio)

        rng = np.random.default_rng(42)
        perm = rng.permutation(num_total)

        # Indexera direkt med permuteringen
        val_idx = perm[:self.val_size]
        train_idx = perm[self.val_size:]
        self.val_detections, self.val_flips = self.detections[val_idx], self.flips[val_idx]
        self.train_detections, self.train_flips = self.detections[train_idx], self.flips[train_idx]
        if self.validation_ratio == 1: # If we validate on the whole set, also train on something
            self.train_detections, self.train_flips = self.val_detections, self.val_flips
    
    def _error_correlation_matrix_full(self) -> np.ndarray:
        """
        Compute the full correlation matrix from the observed detections.

        Returns
        -------
        pij_matrix : np.ndarray
            A symmetric matrix of error-pairing probabilities between detector events.
        """
        x = self.train_detections.astype(np.float64)  # shape (shots, N)

        # Compute means
        mean_i = x.mean(axis=0)  # shape (N,)
        mean_ij = (x.T @ x) / x.shape[0]  # shape (N, N)

        # Numerator and denominator
        numerator = mean_ij - np.outer(mean_i, mean_i)
        denominator = 1 - 2 * mean_i[:, None] - 2 * mean_i[None, :] + 4 * mean_ij

        with np.errstate(divide='ignore', invalid='ignore'):
            sqrt_term = np.sqrt(1 - 4 * numerator / denominator)
            pij = 0.5 - 0.5 * sqrt_term

        pij = np.where(np.isfinite(pij), pij, 0.0)  # Replace NaNs and infs with 0.0
        np.fill_diagonal(pij, 0.0)  # set diagonal to 0 for clarity

        return pij

    
    def _get_edges(self) -> None:
        """
        Build the matching graph with edges weighted according to the selected weight scheme.

        Constructs both space-like (within a time slice) and time-like (between time slices) edges.
        Edge weights are either uniform or derived from the negative log of correlation coefficients.
        """
        row_len = self.distance - 1
        num_detectors = (self.t + 1) * row_len

        # Compute edge weights
        if self.weight_scheme == 'p_ij':
            error_correlation = self._error_correlation_matrix_full()
            error_correlation[error_correlation <= 0] = 1e-7  # Avoid log(0) or negative weights
            weights = -np.log(error_correlation)
        elif self.weight_scheme == 'uniform':
            weights = np.ones((num_detectors, num_detectors))
        else:
            raise ValueError(f"Unknown weight scheme '{self.weight_scheme}'.")

        # Add space-like edges (horizontal, within each time slice)
        for t_index in range(self.t + 1):
            row_start = t_index * row_len
            row_end = row_start + row_len

            for i in range(row_start, row_end - 1):
                self.matcher.add_edge(
                    i, i + 1,
                    weight=weights[i][i + 1],
                    fault_ids={i % row_len + 1},
                    merge_strategy='replace'
                )

            self.matcher.add_boundary_edge(
                row_start,
                weight=weights[row_start][row_start + 1],
                fault_ids={0},
                merge_strategy='replace'
            )

            self.matcher.add_boundary_edge(
                row_end - 1,
                weight=weights[row_end - 2][row_end - 1],
                fault_ids={row_len},
                merge_strategy='replace'
            )

        # Add time-like edges (vertical, across time slices)
        for t_index in range(self.t):
            for offset in range(row_len):
                i = t_index * row_len + offset
                j = i + row_len
                self.matcher.add_edge(
                    i, j,
                    weight=weights[i][j],
                    merge_strategy='replace'
                )

    def _evaluate_predictions(self) -> float:
        """
        Evaluate decoder accuracy using a validation set.

        Returns
        -------
        logical_accuracy : float
            Logical decoding accuracy, including both trivial and non-trivial shots.
        """

        # Filter out trivial syndromes
        nontrivial = np.any(self.val_detections, axis=1)
        detections_nt = self.val_detections[nontrivial]
        flips_nt = self.val_flips[nontrivial]

        # Decode predictions using MWPM
        predictions = self.matcher.decode_batch(detections_nt)

        actual = flips_nt[:, -1]
        predicted = predictions[:, 0]
        correct = np.sum(actual == predicted)
        trivial_count = np.sum(~nontrivial)

        # Logical accuracy over all validation samples
        print(f"Accuracy (excluding trivial syndromes): {correct/predicted.shape[0]:.5f}, n_errors={predicted.shape[0]-correct}")
        logical_accuracy = (correct + trivial_count) / self.val_size
        logical_accuracy_err = standard_deviation(logical_accuracy, self.val_size)

        return logical_accuracy, logical_accuracy_err

    def decode(self) -> float:
        """
        Full decoding pipeline: load data, construct the graph, run decoding, and return accuracy.

        Returns
        -------
        logical_accuracy : float
            Logical accuracy on the validation set.
        """
        self.train_val_split()
        self._get_edges()
        return self._evaluate_predictions()


if __name__ == "__main__":
    args = Args(t=50, distance=15, simulator_backend=False, val_fraction=0.1, load_distance=None, sub_dir="/turning_the_knob", noise_angle=0.1653)
    decoder = MWPMDecoder(args, weight_scheme="p_ij")
    logical_accuracy, logical_accuracy_err = decoder.decode()
    print(f"Decoder completed with logical accuracy: {logical_accuracy:.5f}")
