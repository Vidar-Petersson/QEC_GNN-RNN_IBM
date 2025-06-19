import os
import re
import json
from pathlib import Path
from typing import Tuple, List
import numpy as np
from qiskit_ibm_runtime import RuntimeDecoder


class IBMSampler:
    """
    Loads detection events and logical flip outcomes from IBM or simulated JSON job data.
    Works with either experimental or simulated (Aer) data.
    """

    def __init__(self, distance: int, t: int, simulated: bool = False):
        """
        Initialize the sampler.

        Args:
            distance (int): Code distance (d) of the QEC code.
            t (int): Number of time steps.
            simulated (bool): Whether to use simulated (Aer) or real IBM data.
        """
        self.simulated = simulated
        self.distance = distance
        self.t = t

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
        job_dir = Path("./jobdata/aer") if self.simulated else Path("./jobdata/ibm")
        pattern = re.compile(rf"_({self.distance})_({self.t - 1})_")

        for filename in os.listdir(job_dir):
            if pattern.search(filename):
                return job_dir, filename

        raise FileNotFoundError(
            f"No file found in '{job_dir}' matching pattern '_{self.distance}_{self.t - 1}_'"
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
        code_distance, t, shots, initial_logical_state = map(int, parts[1:])

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

        if self.simulated:
            counts = data.get_counts()
            syndromes, middle_states, final_state = [], [], []
            for bitstring, freq in counts.items():
                syndrome, middle, final = bitstring.split()
                syndromes.extend([syndrome] * freq)
                middle_states.extend([middle] * freq)
                final_state.extend([final] * freq)
        else:
            data = data[0]  # Experimental jobs are returned as a list
            syndromes = data.data.syndrome.get_bitstrings()
            middle_states = data.data.middle_states.get_bitstrings()
            final_state = data.data.final_state.get_bitstrings()

        # Reverse bit order to match IBM's convention
        syndromes = [s[::-1] for s in syndromes]
        middle_states = [s[::-1] for s in middle_states]
        final_state = [s[::-1] for s in final_state]

        return syndromes, middle_states, final_state

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

        mid_syndrome = np.array([[int(b) for b in s] for s in syndromes], dtype=np.uint8)
        final_syndrome = self._compute_syndrome_differences(final_state)

        return np.concatenate([initial_syndrome, mid_syndrome, final_syndrome], axis=1)

    def _compute_syndrome_differences(self, states: List[str]) -> np.ndarray:
        """
        Computes the parity difference between time t-1 and t.

        Args:
            states (List[str]): Final logical state bitstrings.

        Returns:
            np.ndarray: Final syndrome bits, shape (shots, ancillas)
        """
        arr = np.array([[int(c) for c in s] for s in states], dtype=np.uint8)
        return arr[:, :-1] ^ arr[:, 1:]

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

    def _extract_logical_flips(self, middle_states: List[str], final_state: List[str]) -> np.ndarray:
        """
        Extracts the final logical state as binary classification.

        Returns:
            np.ndarray: Boolean array of shape (shots, t), with flip at last time step.
        """
        logical_states = [a + b for a, b in zip(middle_states, [s[0] for s in final_state])]
        matrix = [
            np.maximum.accumulate(np.fromiter(s, dtype=int)).astype(bool)
            for s in logical_states
        ]
        return matrix

    def load_jobdata(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main entry point: loads detection events and logical flip labels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (detector events, final logical flips)
        """
        syndromes, middle_states, final_state = self._load_json()
        syndrome_matrix = self._get_syndrome_matrix(syndromes, final_state)
        detection_events = self._extract_detection_events(syndrome_matrix)
        logical_flips = self._extract_logical_flips(middle_states, final_state)

        return detection_events, logical_flips


if __name__ == "__main__":
    sampler = IBMSampler(distance=3, t=6, simulated=False)
    detection_events, observable_flips = sampler.load_jobdata()

    print("Detection events shape:", detection_events.shape)
    print("Logical flips shape:", observable_flips.shape)