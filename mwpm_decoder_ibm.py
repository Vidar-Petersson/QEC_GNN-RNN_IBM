import time
import numpy as np
import pymatching
import torch
from torch.utils.data import random_split

from args import Args
from dataloader_ibm import IBMSampler


class MWPMDecoder:
    """
    Minimum Weight Perfect Matching (MWPM) decoder for surface code syndrome data.

    This decoder uses PyMatching to construct a matching graph based on observed
    detection events ("syndromes") from a quantum circuit, and predicts logical errors
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
        self.t = args.t[0] - 1  # Number of rounds minus one (for indexing)
        self.simulator_backend = args.simulator_backend
        self.validation_ratio = args.val_fraction
        self.weight_scheme = weight_scheme

        self.matcher = pymatching.Matching()
        self.sampler = IBMSampler(
            distance=self.distance,
            t=self.t + 1,
            simulator=self.simulator_backend
        )

    def _load_job_data(self) -> None:
        """
        Load syndrome and logical flip data from the sampler.
        Also computes a mask identifying trivial (all-zero) syndromes.
        """
        t0 = time.perf_counter()
        self.syndromes, self.flips = self.sampler.load_jobdata()

        self.nontrivial_mask = np.any(self.syndromes, axis=1)
        self.total_shots = self.syndromes.shape[0]
        trivial_share = np.mean(~self.nontrivial_mask)

        print(f"Loaded data '{self.sampler.filename}' (d={self.distance}, t={self.t}) "
              f"with {self.total_shots} shots ({trivial_share*100:.1f}% trivial) "
              f"in {time.perf_counter() - t0:.2f}s.")

    def _error_correlation_matrix(self) -> np.ndarray:
        """
        Compute the error correlation matrix from the observed syndromes.

        Returns
        -------
        correlation_matrix : np.ndarray
            A (N, N) matrix of normalized correlation coefficients between detector events.
            Diagonal elements are set to zero.
        """
        # Marginal detection probabilities for each detector
        marginal_probs = self.syndromes.mean(axis=0)

        # Joint detection probabilities
        joint_probs = (self.syndromes[:, :, None] * self.syndromes[:, None, :]).mean(axis=0)

        # Compute normalized correlations
        denom = np.outer(1 - 2 * marginal_probs, 1 - 2 * marginal_probs)
        numer = joint_probs - np.outer(marginal_probs, marginal_probs)

        with np.errstate(divide='ignore', invalid='ignore'):
            correlation_matrix = np.where(denom != 0, numer / denom, np.inf)

        np.fill_diagonal(correlation_matrix, 0.0)
        return correlation_matrix

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
            error_correlation = self._error_correlation_matrix()
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
        total_samples = self.syndromes.shape[0]
        val_count = int(self.validation_ratio * total_samples)
        train_count = total_samples - val_count

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

        # Decode predictions using MWPM
        predictions = self.matcher.decode_batch(syndromes_nt)

        actual = flips_nt[:, -1]
        predicted = predictions[:, 0]
        correct = np.sum(actual == predicted)
        trivial_count = np.sum(~nontrivial)

        # Logical accuracy over all validation samples
        logical_accuracy = (correct + trivial_count) / val_count
        return logical_accuracy

    def decode(self) -> float:
        """
        Full decoding pipeline: load data, construct the graph, run decoding, and return accuracy.

        Returns
        -------
        logical_accuracy : float
            Logical accuracy on the validation set.
        """
        self._load_job_data()
        self._get_edges()
        return self._evaluate_predictions()


if __name__ == "__main__":
    args = Args(t=[6], distance=3, sliding=True, dt=2, simulator_backend=False)
    decoder = MWPMDecoder(args, weight_scheme="uniform")
    accuracy = decoder.decode()
    print(f"Decoder completed with logical accuracy: {accuracy:.3f}")
