import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))  # Add parent dir for imports

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
        self.t = args.t
        self.noise_angle = round(args.noise_angle, 4)

        self.sub_dir = args.sub_dir
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
        sub_dir = "/"+self.sub_dir if self.sub_dir is not None else ""
        job_dir = Path(f"./jobdata/aer{sub_dir}") if self.simulator else Path(f"./jobdata/ibm{sub_dir}")
        # TODO fix this pattern match, currently you have to change the number os shots to match the desired file
        pattern = re.compile(rf"_({self.load_distance})_({self.t - 1})_(\d+)_0_({self.noise_angle})")

        for filename in os.listdir(job_dir):
            if pattern.search(filename):
                return job_dir, filename

        raise FileNotFoundError(
            f"No file found in '{job_dir}' matching pattern '_{self.load_distance}_{self.t - 1}_20000_0_{self.noise_angle}'"
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
        try: # For backwards compability
            noise_angle = map(float, parts[5])
        except:
            noise_angle = 0.0

        return {
            "file_name": name,
            "job_id": job_id,
            "code_distance": code_distance,
            "ancillas": code_distance - 1,
            "t": t,
            "shots": shots,
            "initial_logical_state": initial_logical_state,
            "noise_angle": noise_angle,
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

            if len(data.data.keys()) > 4: #New repetition_code qiskit-qec
                # 1) Separera ut code_bit och bygg listan av arrayer
                arrays = []
                for name, reg in data.data.items():
                    bits = reg.get_bitstrings()
                    if name == "code_bit":
                        final_state = self._bitstrings_to_array(bits)
                    else:
                        arrays.append(self._bitstrings_to_array(bits))

                # 2) Skapa en 3D-array med form (R, S, Q):
                #    R = antal register utom code_bit
                #    S = antal upprepningar (shots)
                #    Q = antal kvantbitar per register
                regs = np.stack(arrays[::-1], axis=0)  # vänder ordningen och staplar

                # 3) Beräkna kumulativa syndrom-bitar (flipp/icke-flipp)
                #    diff[i] = regs[i] != regs[i+1], form (R-1, S, Q)
                diff = (regs[:-1] != regs[1:]).astype(np.uint8)

                # 4) Lägg till sista registret oförändrat som sista skikt
                last = regs[-1:].astype(np.uint8)  # form (1, S, Q)

                # 5) Kombinera till syndrom-array form (R, S, Q)
                syndrome_stack = np.concatenate([diff, last], axis=0)

                # 6) Permutera så att första dimensionen är shots (S), sedan register×qubits
                #    och sluttligen "flattenar" de två sista till en 2D-array
                shots = syndrome_stack.shape[1]
                R, _, Q = syndrome_stack.shape
                syndromes = syndrome_stack.transpose(1, 0, 2).reshape(shots, R * Q)

            else: # Old code
                syndromes = self._bitstrings_to_array(data.data.syndromes.get_bitstrings())
                final_state = self._bitstrings_to_array(data.data.final_state.get_bitstrings())

        if hasattr(data.data, "middle_states"):
            middle_states = self._bitstrings_to_array(data.data.middle_states.get_bitstrings())
        else:
            middle_states = None
            print("Warning: Jobdata doesn't include middle_states!")
        

        # Reverse bit order to match IBM's convention
        syndromes = syndromes[:, ::-1]
        if middle_states is not None:
            middle_states = middle_states[:, ::-1]
        final_state = final_state[:, ::-1]

        return syndromes, middle_states, final_state

    def _get_syndrome_matrix(self, mid_syndromes: List[str], final_state: List[str]) -> np.ndarray:
        """
        Builds the full syndrome matrix including initial and final logical readings.
        Returns:
            np.ndarray: Shape (shots, ancillas * time_steps)
        """
        ancillas = self.job_params["ancillas"]
        shots = self.job_params["shots"]
        init_bit = str(self.job_params["initial_logical_state"])
        initial_syndrome = np.full((shots, ancillas), int(init_bit), dtype=np.uint8)

        final_syndrome = final_state[:, :-1] ^ final_state[:, 1:]

        return np.concatenate([initial_syndrome, mid_syndromes, final_syndrome], axis=1)

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

        shots, num_logicals = final_state.shape

        if logical_index is None:
            flips = final_state == 1  # shape: (shots, num_logicals)
            matrix = np.zeros((shots, num_logicals, self.t), dtype=bool)
            matrix[:, :, -1] = flips
            return matrix  # shape: (shots, num_logicals, t)
        else:
            flips = final_state[:, logical_index] == 1
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

        if sub_det.shape[0] > 1_00_000: # Esnures maximum of 1 million shots per configuration
            # Add random seed!
            row_index = np.random.choice(sub_det.shape[0], size=1_00_000, replace=False)
            sub_det   = sub_det[row_index]
            sub_flips = sub_flips[row_index]

        return sub_det, sub_flips
    
    @staticmethod
    def _bitstrings_to_array(bitstrings: List[str]) -> np.ndarray:
        return np.frombuffer(''.join(bitstrings).encode(), dtype='S1').view(np.uint8).reshape(len(bitstrings), -1) - ord('0')


if __name__ == "__main__":
    args = Args(t=[50], distance=49, noise_angle=0.2513, simulator_backend=False)
    sampler = IBMSampler(args)
    detection_events, observable_flips = sampler.load_jobdata(verbose=True)
    print("Original detection events and logical flips shape:", detection_events.shape)
    print("original Logical flips shape:", observable_flips.shape)