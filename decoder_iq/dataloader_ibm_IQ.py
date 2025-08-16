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

from decoder_iq.GMM_classifier import GMMClassifier
import matplotlib.pyplot as plt

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
        self.initial_state = 0
        self.load_distance = args.load_distance if args.load_distance is not None else args.distance
        self.t = args.t
        self.noise_angle = round(args.noise_angle, 4)
        self.detection_threshold = args.detection_threshold
        self.seed = args.seed

        self.sub_dir = args.sub_dir
        self.job_dir, self.filenames = self._find_filenames()
        self.job_params = self._parse_job_params(self.filenames[0])

    def _find_filenames(self) -> Tuple[Path, list[str]]:
        """
        Finds all job files that match the code distance, time steps and noise angle.
        Returns:
            Tuple[Path, List[str]]: The job directory and a list of matching filenames.
        Raises:
            FileNotFoundError: If no matching files are found.
        """
        sub_dir = self.sub_dir if self.sub_dir is not None else ""
        job_dir = Path(f"./jobdata/aer{sub_dir}") if self.simulator else Path(f"./jobdata/ibm{sub_dir}")

        # Matcha alla shots (en eller flera siffror)
        pattern = re.compile(rf"_({self.load_distance})_({self.t - 1})_(\d+)_({self.initial_state})_({self.noise_angle})")

        matching_files = [f for f in os.listdir(job_dir) if pattern.search(f)]
        # matching_files = matching_files[:1]

        if not matching_files:
            raise FileNotFoundError(
                f"No files found in '{job_dir}' matching pattern '_{self.load_distance}_{self.t - 1}_<shots>_0_{self.noise_angle}'"
            )
        else:
            print(f"Found {len(matching_files)} files matching pattern '_{self.load_distance}_{self.t - 1}_<shots>_0_{self.noise_angle}'")

        return job_dir, matching_files

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

    def _load_json(self):
        """
        Loads and concatenates syndrome and final logical state data from all matching JSON files.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (syndromes_soft, middle_states, final_state_soft)
        """
        all_syndromes = []
        all_final_states = []

        for filename in self.filenames:  # self.filenames sätts i __init__
            job_path = self.job_dir / filename
            with open(job_path) as f:
                data = json.load(f, cls=RuntimeDecoder)

            if self.simulator:
                print("Simulator has no ability to produce IQ-data")
                continue

            if "code_bit" in data[0].data.keys():
                assert data[0].data.code_bit.dtype == "complex128"

                raw_arrays = []
                for name, reg in data[0].data.items():
                    if name == "code_bit":
                        final_state_raw = np.array(reg)
                        final_state_raw = np.expand_dims(final_state_raw[:, ::-1], axis=0)
                    else:
                        raw_arrays.append(reg)
                raw_arrays = np.stack(raw_arrays, axis=0)[:, :, ::-1]

                self.syndrome_calibrator = GMMClassifier(calibration_data=raw_arrays)
                state_prob = self.syndrome_calibrator.compute_p_state(raw_arrays)
                syndromes_soft = self._reset_adjust(state_prob)

                self.final_state_calibrator = GMMClassifier(calibration_data=final_state_raw)
                final_state_soft = self.final_state_calibrator.compute_p_state(final_state_raw)[0]

                all_syndromes.append(syndromes_soft)
                all_final_states.append(final_state_soft)
            print(f"Loaded IQ-data from: {filename}")

        # Slå ihop längs shots-axeln
        syndromes_concat = np.concatenate(all_syndromes, axis=0)
        final_states_concat = np.concatenate(all_final_states, axis=0)

        return syndromes_concat, None, final_states_concat
    
    def plot_flip_distribution(self, flip_p, detector_idx=None, bins=50):
        """
        Plotta fördelningen av flip-sannolikheter.
        
        flip_p: np.ndarray shape (S, R-1, D)
        detector_idx: int eller None. Om None plottas alla detektorer ihop.
        bins: antal histogram-bins.
        """
        ancillas = self.job_params["ancillas"]
        T = flip_p.shape[1] // ancillas
        # Reshape till (shots, T, ancillas)
        resh = flip_p.reshape(-1, T, ancillas)
        if detector_idx is not None:
            data = resh[:, :, detector_idx].ravel()
            title = f"Flip probability distribution - Detector {detector_idx}"
        else:
            data = resh.ravel()
            title = "Flip probability distribution - All detectors"

        plt.figure(figsize=(6,4))
        plt.hist(data, bins=bins, density=True, alpha=0.6, color='tab:blue', label='Histogram')

        # try:
        #     from scipy.stats import gaussian_kde
        #     kde = gaussian_kde(data)
        #     x_vals = np.linspace(0, 1, 500)
        #     plt.plot(x_vals, kde(x_vals), color='black', lw=1.5, label='KDE')
        # except ImportError:
        #     pass  # Om scipy inte finns, hoppa över KDE

        plt.xlabel("Flip probability")
        plt.ylabel("Density")
        plt.yscale("log")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _reset_adjust(self, no_reset_soft):
        """
        Adjusts syndromes when reset=False is used in a repetition code.

        Computes bitwise differences between successive rounds (left-to-right)
        to infer detection events. Optionally computes soft (probabilistic)
        syndromes if soft measurement data is provided.

        Parameters
        ----------
        no_reset_soft : Optional[np.ndarray], default=None
            Soft values representing P(measured=1), same shape as `no_reset`.

        Returns
        -------
        syndromes_soft : Optional[np.ndarray]
            Soft syndromes of shape (S, R * Q), or None.
        """

        # Soft XOR diff (left to right)
        p_diff = no_reset_soft[1:] * (1 - no_reset_soft[:-1]) + (1 - no_reset_soft[1:]) * no_reset_soft[:-1]
        p_first = no_reset_soft[0:1]
        syndrome_soft_stack = np.concatenate([p_first, p_diff], axis=0)

        # Reshape to (S, R*Q)
        R, S, Q = syndrome_soft_stack.shape
        syndromes_soft = syndrome_soft_stack.transpose(1, 0, 2).reshape(S, R * Q)

        return syndromes_soft

    def _get_syndrome_matrix_soft(self, mid_syndromes_soft: List[str], final_state_soft: List[str]) -> np.ndarray:
        """
        Builds the full syndrome matrix including initial and final logical readings.
        Returns:
            np.ndarray: Shape (shots, ancillas * time_steps)
        """
        ancillas = self.job_params["ancillas"]
        shots = mid_syndromes_soft.shape[0]
        init_bit = str(self.job_params["initial_logical_state"])

        initial_syndrome_soft = np.full((shots, ancillas), int(init_bit), dtype=np.uint8)

        final_syndrome_soft = final_state_soft[:,:-1] * (1 - final_state_soft[:,1:]) + (1 - final_state_soft[:,:-1]) * final_state_soft[:,1:]
        
        return np.concatenate([initial_syndrome_soft, mid_syndromes_soft, final_syndrome_soft], axis=1)
    
    def _extract_detection_event_probs(self, syndrome_soft: np.ndarray) -> np.ndarray:
        """
        syndrome_soft: np.array shape (shots, ancillas * time_steps)
        Returnerar en matris shape (shots, ancillas*(time_steps-1))
        med P(det_evt=1) för varje position.
        """
        ancillas = self.job_params["ancillas"]
        T = syndrome_soft.shape[1] // ancillas
        # Reshape till (shots, T, ancillas)
        resh = syndrome_soft.reshape(-1, T, ancillas)
        # XOR‐sannolikhet längs tidsaxeln:
        # P(flip@t) = p_t*(1-p_{t+1}) + (1-p_t)*p_{t+1}
        p_t   = resh[:, :-1, :]
        p_tp1 = resh[:,  1:, :]
        p_flip = p_t*(1-p_tp1) + (1-p_t)*p_tp1  # → (shots, T-1, ancillas)
        # Platta ut till (shots, ancillas*(T-1))
        return p_flip.reshape(resh.shape[0], -1)
        
    def _extract_logical_flip_probs(self, final_state_soft: np.ndarray, logical_index: int | None = 0) -> np.ndarray:
        """
        final_state_soft: sannolikhet att det uppmätta tillståndet är 1, shape (shots, distance)
        initial_logical_state: 0 eller 1 (från job_params)
        Returnerar, sannolikhet att en flipp skett från initialiseringen, shape (shots,)
        """
        init = self.job_params["initial_logical_state"]
        shots, num_logicals = final_state_soft.shape

        if logical_index is None:
            if init == 0:
                flips = final_state_soft
            else:
                flips = 1 - final_state_soft
        else:
            if init == 0:
                flips = final_state_soft[:,logical_index]
            else:
                flips = 1 - final_state_soft[:,logical_index]

        if logical_index is None:
            matrix = np.zeros((shots, num_logicals, self.t))
            matrix[:, :, -1] = flips
            return matrix  # shape: (shots, num_logicals, t)
        else:
            matrix = np.zeros((shots, self.t))
            matrix[:, -1] = flips
            return matrix  # shape: (shots, t)
        # Om init=0 är P(flip)=p_final, annars P(flip)=1-p_final


    def load_jobdata(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main entry point: loads detection events and logical flip labels.
        Returns:
            Tuple[np.ndarray, np.ndarray]: (detector events, final logical flips)
        """
        t0 = time.perf_counter()
        syndromes_soft, middle_states, final_state_soft = self._load_json()
        # Bygg hårda och mjuka syndrommatriser
        syndrome_soft_matrix = self._get_syndrome_matrix_soft(syndromes_soft, final_state_soft)
        # Mjuka det‐events (sannolikheter)
        detection_events_probs = self._extract_detection_event_probs(syndrome_soft_matrix)

        self.plot_flip_distribution(detection_events_probs)

        trivial_share_before = np.mean(~np.any((detection_events_probs >= self.detection_threshold).astype(int), axis=1)) # Calculates share of trivial graphs

        if self.load_distance == self.distance:
            logical_flips_probs = self._extract_logical_flip_probs(final_state_soft, logical_index=0) # Mjuka logiska flips

        else: # Here we use sampler to downsample to lower distances
            logical_flips_probs = self._extract_logical_flip_probs(final_state_soft, logical_index=None)

            detection_events_probs, logical_flips_probs = self.subsampler(detection_events_probs, logical_flips_probs)
            trivial_share_after = np.mean(~np.any((detection_events_probs >= self.detection_threshold).astype(int), axis=1))

        if verbose:
            print("------------------------------------------------------------------------")
            print(f"Loaded jobdata '{self.filenames}' (d={self.load_distance}, t={self.t}) "
            f"with {syndromes_soft.shape[0]} shots ({trivial_share_before*100:.1f}% trivial).", end=' ')


            if self.load_distance != self.distance:
                print(f"\nSubsampled to d={self.distance} with {detection_events_probs.shape[0]} shots ({trivial_share_after*100:.1f}% trivial with threshold at {self.detection_threshold:.1f}).", end=' ')
            print(f"Total time: {time.perf_counter() - t0:.2f}s.")
            print("------------------------------------------------------------------------")

        return detection_events_probs, logical_flips_probs


    def subsampler(self, det_full: np.ndarray, logical_flips_all) -> Tuple[np.ndarray, np.ndarray]:
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

        if sub_det.shape[0] > 1_000_000: # Ensures maximum of 1 million shots per configuration
            print(f"Reducing number of shots to 1_000_000")
            rng = np.random.default_rng(seed=self.seed)
            row_index = rng.choice(sub_det.shape[0], size=1_000_000, replace=False)
            sub_det   = sub_det[row_index]
            sub_flips = sub_flips[row_index]

        return sub_det, sub_flips
    
    @staticmethod
    def _bitstrings_to_array(bitstrings: List[str]) -> np.ndarray:
        return np.frombuffer(''.join(bitstrings).encode(), dtype='S1').view(np.uint8).reshape(len(bitstrings), -1) - ord('0')


if __name__ == "__main__":
    args = Args(t=51, distance=15, noise_angle=0.0, simulator_backend=False, load_distance=None, sub_dir="/iq_data/training_data")
    sampler = IBMSampler(args)
    detections, flips, detections_probs, flips_probs = sampler.load_jobdata(verbose=True)
    print("Original detection events and soft shape:", detections.shape, detections_probs.shape)
    print("original Logical flips shape:", flips.shape, flips_probs.shape)