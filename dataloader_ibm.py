import os
import re
import json
from pathlib import Path
from typing import Tuple, List
import numpy as np
import time

from args import Args
from qiskit_ibm_runtime import RuntimeDecoder


class IBMSampler:
    """
    Loads detection events and logical flip outcomes from IBM or simulator JSON job data.
    Works with either experimental or simulator (Aer) data.
    """

    def __init__(self, args: Args):
        """
        Initialize the sampler.
        """

        self.simulator = args.simulator_backend
        self.distance = args.distance
        self.load_distance = args.load_distance if args.load_distance is not None else args.distance
        self.t = args.t[0]

        self.job_dir, self.filename = self._find_filename()
        self.job_params = self._parse_job_params(self.filename)

    def _find_filename(self) -> Tuple[Path, str]:
        """
        Finds a job file that matches the code distance and time steps.
        Returns:
            Tuple[Path, str]: The job directory and matching filename.
        Raises:
            FileNotFoundError: If no matching file is found.
        """
        job_dir = Path("./jobdata/aer") if self.simulator else Path("./jobdata/ibm")
        pattern = re.compile(rf"_({self.load_distance})_({self.t - 1})_")

        for filename in os.listdir(job_dir):
            if pattern.search(filename):
                return job_dir, filename

        raise FileNotFoundError(
            f"No file found in '{job_dir}' matching pattern '_{self.load_distance}_{self.t - 1}_'"
        )

    def _parse_job_params(self, filename: str) -> dict:
        """
        Parses job parameters from the filename.
        Args:
            filename (str): Job filename.
        Returns:
            dict: Parsed job parameters.
        """
        name = Path(filename).stem  # Remove extension
        parts = name.split("_")

        job_id = parts[0]
        code_distance, t, shots, initial_logical_state = map(int, parts[1:5])

        return {
            "file_name": name,
            "job_id": job_id,
            "code_distance": code_distance,
            "ancillas": code_distance - 1,
            "t": t,
            "shots": shots,
            "initial_logical_state": initial_logical_state,
        }

    def _load_json(self) -> Tuple[List[str], List[str]]:
        """
        Loads syndrome and final logical state data from JSON file.
        Returns:
            Tuple[List[str], List[str]]: (syndrome bitstrings, final state bitstrings)
        """
        job_path = self.job_dir / self.filename

        with open(job_path) as f:
            data = json.load(f, cls=RuntimeDecoder)

        if self.simulator:
            counts = data.get_counts()
            syndromes, middle_states, final_state = [], [], []
            for bitstring, freq in counts.items():
                syndrome, middle, final = bitstring.split()
                syndromes.extend([syndrome] * freq)
                middle_states.extend([middle] * freq)
                final_state.extend([final] * freq)
        else:
            data = data[0]  # Experimental jobs are returned as a list
            syndromes = data.data.syndromes.get_bitstrings()
            if hasattr(data.data, "middle_states"):
                middle_states = data.data.middle_states.get_bitstrings()
            else:
                middle_states = None
                print("Warning: Jobdata doesn't include middle_states!")
            final_state = data.data.final_state.get_bitstrings()

        # Reverse bit order to match IBM's convention
        syndromes = [s[::-1] for s in syndromes]
        if middle_states is not None:
            middle_states = [s[::-1] for s in middle_states]
        final_state = [s[::-1] for s in final_state]

        return syndromes, middle_states, final_state

    def _compute_syndrome_differences(self, states: List[str]) -> np.ndarray:
        """
        Computes the parity difference between time t-1 and t.
        Args:
            states (List[str]): Final logical state bitstrings.
        Returns:
            np.ndarray: Final syndrome bits, shape (shots, ancillas)
        """
        arr = self._bitstrings_to_array(states)
        return arr[:, :-1] ^ arr[:, 1:]

    def _get_syndrome_matrix(self, syndromes: List[str], final_state: List[str]) -> np.ndarray:
        """
        Builds the full syndrome matrix including initial and final logical readings.
        Returns:
            np.ndarray: Shape (shots, ancillas * time_steps)
        """
        ancillas = self.job_params["ancillas"]
        shots = self.job_params["shots"]
        init_bit = str(self.job_params["initial_logical_state"])
        initial_syndrome = np.full((shots, ancillas), int(init_bit), dtype=np.uint8)

        mid_syndrome = self._bitstrings_to_array(syndromes)
        final_syndrome = self._compute_syndrome_differences(final_state)

        return np.concatenate([initial_syndrome, mid_syndrome, final_syndrome], axis=1)

    def _extract_detection_events(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Converts syndrome matrix to detection event matrix (flips).
        Returns:
            np.ndarray: Boolean matrix of shape (shots, ancillas * (t - 1))
        """
        ancillas = self.job_params["ancillas"]
        T = syndrome.shape[1] // ancillas
        reshaped = syndrome.reshape(-1, T, ancillas)
        flips = np.diff(reshaped, axis=1).astype(bool)
        return flips.reshape(flips.shape[0], -1)

    def _extract_logical_flips(self, middle_states: List[str], final_state: List[str], logical_index: int | None = 0) -> np.ndarray:
        """
        Extracts the final logical state(s) as binary classification.
        If logical_index=None, returns shape (shots, num_logical_qubits, t)
        """
        if middle_states is not None:
            print("Warning: Middle state handling not yet implemented")

        final_array = self._bitstrings_to_array(final_state)  # shape (shots, num_logical_qubits)
        shots, num_logicals = final_array.shape

        if logical_index is None:
            flips = final_array == 1  # shape: (shots, num_logicals)
            matrix = np.zeros((shots, num_logicals, self.t), dtype=bool)
            matrix[:, :, -1] = flips
            return matrix  # shape: (shots, num_logicals, t)
        else:
            flips = final_array[:, logical_index] == 1
            matrix = np.zeros((shots, self.t), dtype=bool)
            matrix[:, -1] = flips
            return matrix  # shape: (shots, t)

    def load_jobdata(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main entry point: loads detection events and logical flip labels.
        Returns:
            Tuple[np.ndarray, np.ndarray]: (detector events, final logical flips)
        """
        t0 = time.perf_counter()
        syndromes, middle_states, final_state = self._load_json()
        syndrome_matrix = self._get_syndrome_matrix(syndromes, final_state)
        detection_events = self._extract_detection_events(syndrome_matrix)
        trivial_share_before = np.mean(~np.any(detection_events, axis=1))

        if self.load_distance == self.distance:
            logical_flips = self._extract_logical_flips(middle_states, final_state, logical_index=0)

        else:
            logical_flips_all = self._extract_logical_flips(middle_states, final_state, logical_index=None)
            detection_events, logical_flips = self.subsampler(detection_events, logical_flips_all)
            trivial_share_after = np.mean(~np.any(detection_events, axis=1))

        if verbose:
            print("------------------------------------------------------------------------")
            print(f"Loaded jobdata '{self.filename}' (d={self.load_distance}, t={self.t}) "
            f"with {len(syndromes)} shots ({trivial_share_before*100:.1f}% trivial).", end=' ')


            if self.load_distance != self.distance:
                print(f"\nSubsampled to d={self.distance} with {detection_events.shape[0]} shots ({trivial_share_after*100:.1f}% trivial).", end=' ')
            print(f"Total time: {time.perf_counter() - t0:.2f}s.")
            print("------------------------------------------------------------------------")

        return detection_events, logical_flips

    
    def subsampler(self, det_full: np.ndarray, logical_flips_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Efficient subsampling of detection events and corresponding logical flips.
        Args:
            det_full (np.ndarray): Detection events, shape (shots, ancillas*(t-1))
            logical_flips_all (np.ndarray): shape (shots, num_logical_qubits, t)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Subsampled detection events and logical flips.
        """

        shots, total_events = det_full.shape
        full_anc = self.load_distance - 1
        target_anc = self.distance - 1
        steps = total_events // full_anc

        det_reshaped = det_full.reshape(shots, steps, full_anc)

        subsampled_dets = []
        subsampled_flips = []

        for start in range(full_anc - target_anc + 1):
            window = det_reshaped[:, :, start : start + target_anc]
            subsampled_dets.append(window.reshape(shots, -1))

            flips_for_window = logical_flips_all[:, start, :]  # shape: (shots, t)
            subsampled_flips.append(flips_for_window)

        sub_det = np.vstack(subsampled_dets)
        sub_flips = np.vstack(subsampled_flips)

        return sub_det[:100000], sub_flips[:100000]
    
    @staticmethod
    def _bitstrings_to_array(bitstrings: List[str]) -> np.ndarray:
        return np.frombuffer(''.join(bitstrings).encode(), dtype='S1').view(np.uint8).reshape(len(bitstrings), -1) - ord('0')


if __name__ == "__main__":
    args = Args(t=[6], distance=3, simulator_backend=False, load_distance=5)
    sampler = IBMSampler(args)
    detection_events, observable_flips = sampler.load_jobdata(verbose=True)
    print("Original detection events and logical flips shape:", detection_events.shape)
    print("original Logical flips shape:", observable_flips.shape)