import os
import re
import json
from pathlib import Path
from typing import Tuple, List
import numpy as np
import time

from args import Args
from qiskit_ibm_runtime import RuntimeDecoder

from IQ_to_psoft import SoftCalibrator

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
        pattern = re.compile(rf"_({self.load_distance})_({self.t - 1})_20000_0_{self.noise_angle}")

        for filename in os.listdir(job_dir):
            if pattern.search(filename):
                return job_dir, filename

        raise FileNotFoundError(
            f"No file found in '{job_dir}' matching pattern '_{self.load_distance}_{self.t - 1}'"
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

        if self.simulator: # Simulator has no ability to produce IQ-data
            counts = data.get_counts()
            syndromes, middle_states, final_state = [], [], []
            for bitstring, freq in counts.items():
                syndrome, middle, final = bitstring.split()
                syndromes.extend([syndrome] * freq)
                middle_states.extend([middle] * freq)
                final_state.extend([final] * freq)
        else:
            if "code_bit" in data[0].data.keys(): # New repetition code

                if data[0].data.code_bit.dtype == "complex128": # if IQ-output from rep.code
                    raw_arrays = []
                    for name, reg in data[0].data.items():
                        if name == "code_bit":
                            final_state_raw = reg
                            final_state_raw = np.expand_dims(final_state_raw, axis=0) # Input formatting to the SoftCalibrator class
                        else:
                            raw_arrays.append(reg)

                    raw_arrays = np.stack(raw_arrays[::-1], axis=0) 

                    self.syndrome_calibrator = SoftCalibrator(calibration_data=raw_arrays)
                    no_reset_soft = self.syndrome_calibrator.compute_p_soft(raw_arrays)
                    no_reset = self.syndrome_calibrator.infer_hard(raw_arrays)
                    syndromes, syndromes_soft = self._reset_adjust(no_reset, no_reset_soft)
                    #self.syndrome_calibrator.visualize_iq_with_psoft(raw_arrays, no_reset_soft, detector_index=1)


                    self.final_state_calibrator = SoftCalibrator(calibration_data=final_state_raw)
                    final_state_soft = self.final_state_calibrator.compute_p_soft(final_state_raw)[0]
                    final_state = self.final_state_calibrator.infer_hard(final_state_raw)[0]
                    final_state_soft = final_state * (1 - final_state_soft) + (1 - final_state) * final_state_soft # convert to P(b=1), probability of being one
                    #self.final_state_calibrator.visualize_iq_with_psoft(final_state_raw, final_state_soft, detector_index=0)

                else: #New repetition_code qiskit-qec
                    # 1) Separera ut code_bit och bygg listan av arrayer
                    raw_arrays = []
                    for name, reg in data[0].data.items():
                        bits = reg.get_bitstrings()
                        if name == "code_bit":
                            final_state = self._bitstrings_to_array(bits)
                        else:
                            raw_arrays.append(self._bitstrings_to_array(bits))
                    raw_arrays = np.stack(raw_arrays[::-1], axis=0)  # Skapa en 3D-array med form (Rundor, Shots, Detektorer), right to left
                    syndromes = self._reset_adjust(raw_arrays)

            else: # Old repetition code
                syndromes = self._bitstrings_to_array(data.data.syndromes.get_bitstrings())
                final_state = self._bitstrings_to_array(data.data.final_state.get_bitstrings())

        if hasattr(data[0].data, "middle_states"):
            middle_states = self._bitstrings_to_array(data.data.middle_states.get_bitstrings())
        else:
            middle_states = None
            print("Warning: Jobdata doesn't include middle_states!")
        
        # Reverse bit order, IBM's convention is right -> left read-out
        syndromes = syndromes[:, ::-1]
        syndromes_soft = syndromes_soft[:, ::-1]
        if middle_states is not None:
            middle_states = middle_states[:, ::-1]
        final_state = final_state[:, ::-1]
        final_state_soft = final_state_soft[:, ::-1]

        return syndromes, syndromes_soft, middle_states, final_state, final_state_soft

    def _reset_adjust(self, no_reset, no_reset_soft = None):
        """"
        If reset=False is used in the repetition code, an adjustment is needed
        also propagates the adjustment for the soft info.
        """

        # Beräkna kumulativa syndrom-bitar (flipp/icke-flipp)
        #    diff[i] = regs[i] != regs[i+1], form (R-1, S, Q)
        diff = (no_reset[:-1] != no_reset[1:]).astype(np.uint8)

        # Lägg till sista registret oförändrat som sista skikt
        last = no_reset[-1:].astype(np.uint8)  # form (1, S, Q)

        # Kombinera till syndrom-array form (R, S, Q)
        syndrome_stack = np.concatenate([diff, last], axis=0)

        # Permutera så att första dimensionen är shots (S), sedan register×qubits
        #    och sluttligen "flattenar" de två sista till en 2D-array
        shots = syndrome_stack.shape[1]
        R, _, Q = syndrome_stack.shape
        syndromes = syndrome_stack.transpose(1, 0, 2).reshape(shots, R * Q)

        if no_reset_soft is not None:
            # 1) Beräkna p = P(verkligt 1)
            p = no_reset * (1 - no_reset_soft) + (1 - no_reset) * no_reset_soft

            # 2) “Mjukt diff” precis som XOR men med sannolikheter
            #    P(XOR=1) = p_t*(1-p_{t+1}) + (1-p_t)*p_{t+1}
            p_diff = p[:-1] * (1 - p[1:]) + (1 - p[:-1]) * p[1:]   # → form (R-1, S, Q)

            # 3) Ta med sista laget (oförändrat, dvs bär över sista p)
            p_last = p[-1:]                                        # → form (1, S, Q)

            # 4) Bygg upp “soft_syndrome_stack” och reshapa till (S, R*Q)
            syndrome_soft_stack = np.concatenate([p_diff, p_last], axis=0)  # (R, S, Q)
            R, S, Q = syndrome_soft_stack.shape

            syndromes_soft = (
                syndrome_soft_stack
                .transpose(1, 0, 2)    # → (S, R, Q)
                .reshape(S, R * Q)     # → (S, R*Q)
            )
            return syndromes, syndromes_soft
        return syndromes

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

    def _get_syndrome_matrix_soft(self, mid_syndromes_soft: List[str], final_state, final_state_soft: List[str]) -> np.ndarray:
        """
        Builds the full syndrome matrix including initial and final logical readings.
        Returns:
            np.ndarray: Shape (shots, ancillas * time_steps)
        """
        ancillas = self.job_params["ancillas"]
        shots = self.job_params["shots"]
        init_bit = str(self.job_params["initial_logical_state"])

        initial_syndrome_soft = np.full((shots, ancillas), int(init_bit), dtype=np.uint8)

        final_syndrome_soft = final_state_soft[:,:-1] * (1 - final_state_soft[:,1:]) + (1 - final_state_soft[:,:-1]) * final_state_soft[:,1:]
        
        return np.concatenate([initial_syndrome_soft, mid_syndromes_soft, final_syndrome_soft], axis=1)

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
        
    def _extract_logical_flip_probs(self, final_state_soft: np.ndarray, logical_index) -> np.ndarray:
        """
        final_state_soft: shape (shots, distance)
        initial_logical_state: 0 eller 1 (från job_params)
        Returnerar shape (shots,) med P(logical_flip=1)
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
        syndromes, syndromes_soft, middle_states, final_state, final_state_soft = self._load_json()

        # Bygg hårda och mjuka syndrommatriser
        syndrome_matrix = self._get_syndrome_matrix(syndromes, final_state)
        syndrome_soft_matrix = self._get_syndrome_matrix_soft(syndromes_soft, final_state, final_state_soft)

        # Hårda det‐events
        detection_events      = self._extract_detection_events(syndrome_matrix)
        trivial_share_before = np.mean(~np.any(detection_events, axis=1))
        # Mjuka det‐events (sannolikheter)
        detection_event_probs = self._extract_detection_event_probs(syndrome_soft_matrix)


        if self.load_distance == self.distance:
            # Hårda logiska flips
            logical_flips      = self._extract_logical_flips(middle_states, final_state, logical_index=0)
            # Mjuka logiska flips
            logical_flips_probs = self._extract_logical_flip_probs(final_state_soft, logical_index=0)

        else: # Here we use sampler to downsample to lower distances
            print(final_state.shape)
            print(final_state_soft.shape)
            logical_flips_all = self._extract_logical_flips(middle_states, final_state, logical_index=None)
            logical_flips_probs = self._extract_logical_flip_probs(final_state_soft, logical_index=None)

            detection_events, detection_event_probs, logical_flips, logical_flips_probs = self.subsampler(detection_events, detection_event_probs, logical_flips_all, logical_flips_probs)
            trivial_share_after = np.mean(~np.any(detection_events, axis=1))

        if verbose:
            print("------------------------------------------------------------------------")
            print(f"Loaded jobdata '{self.filename}' (d={self.load_distance}, t={self.t}) "
            f"with {len(syndromes)} shots ({trivial_share_before*100:.1f}% trivial).", end=' ')


            if self.load_distance != self.distance:
                print(f"\nSubsampled to d={self.distance} with {detection_events.shape[0]} shots ({trivial_share_after*100:.1f}% trivial).", end=' ')
            print(f"Total time: {time.perf_counter() - t0:.2f}s.")
            print("------------------------------------------------------------------------")

        return detection_events, logical_flips, detection_event_probs, logical_flips_probs


    def subsampler(self, det_full: np.ndarray, det_full_probs: np.ndarray, logical_flips_all: np.ndarray, logical_flips_probs) -> Tuple[np.ndarray, np.ndarray]:
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
        det_probs_reshaped = det_full_probs.reshape(shots, steps, full_anc)

        subsampled_dets = []
        subsampled_dets_probs = []
        subsampled_flips = []
        subsampled_flips_probs = []

        for start in range(full_anc - target_anc + 1):
            window = det_reshaped[:, :, start : start + target_anc]
            window_probs = det_probs_reshaped[:, :, start : start + target_anc]
            subsampled_dets.append(window.reshape(shots, -1))
            subsampled_dets_probs.append(window_probs.reshape(shots, -1))

            flips_for_window = logical_flips_all[:, start, :]  # shape: (shots, t)
            subsampled_flips.append(flips_for_window)
            flips_probs_for_window = logical_flips_probs[:, start, :]  # shape: (shots, t)
            subsampled_flips_probs.append(flips_probs_for_window)

        sub_det = np.vstack(subsampled_dets)
        sub_det_probs = np.vstack(subsampled_dets_probs)
        sub_flips = np.vstack(subsampled_flips)
        sub_flips_probs = np.vstack(subsampled_flips_probs)

        if sub_det.shape[0] > 1_000_000: # Esnures maximum of 1 million shots per configuration
            # Add random seed!
            row_index = np.random.choice(sub_det.shape[0], size=1_000_000, replace=False)
            sub_det   = sub_det[row_index]
            sub_det_probs   = sub_det_probs[row_index]
            sub_flips = sub_flips[row_index]
            sub_flips_probs = sub_flips_probs[row_index]

        return sub_det, sub_det_probs, sub_flips, sub_flips_probs
    
    @staticmethod
    def _bitstrings_to_array(bitstrings: List[str]) -> np.ndarray:
        return np.frombuffer(''.join(bitstrings).encode(), dtype='S1').view(np.uint8).reshape(len(bitstrings), -1) - ord('0')


if __name__ == "__main__":
    args = Args(t=[6], distance=2, noise_angle=0.0, simulator_backend=False, load_distance=3)
    sampler = IBMSampler(args)
    detections, flips, detections_probs, flips_probs = sampler.load_jobdata(verbose=True)
    print("Original detection events and logical flips shape:", detections.shape, detections_probs.shape)
    print("original Logical flips shape:", flips.shape, flips_probs.shape)