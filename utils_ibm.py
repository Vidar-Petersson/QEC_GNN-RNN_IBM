import numpy as np
from pathlib import Path
import os
import re
import json
from qiskit_ibm_runtime import RuntimeDecoder

class IBM_sampler:
    """Class for loading detection_events and observable_flips from existing json data.
    If jobdata doesn't exists for the desired config, ask to generate it via IBM Quantum."""
    
    def __init__(self, distance: int, t: int):
        self.job_dir = Path("./ibm_jobdata/")
        self.distance = distance 
        self.t = t
        self.filename = self._find_filename(self.job_dir, self.distance, self.t-1)
        self.job_params = self._get_job_params(self.filename)
        self.device = "cpu"

    @staticmethod
    def _find_filename(job_dir: str, d: int, t: int) -> str:
        """Find the filename in the given directory that matches the pattern _<d>_<t>_"""
        pattern = re.compile(rf"_({d})_({t})_")
        for filename in os.listdir(job_dir):
            if pattern.search(filename):
                return filename
        raise FileNotFoundError(f"No file found in '{job_dir}' with pattern '_{d}_{t}_'")

    @staticmethod
    def _get_job_params(filename: str) -> dict:
        """Extract job parameters from a job filename."""
        filename = filename.split(".")[0]  # Remove file extension
        filename = filename.split("/")[-1]  # Remove path
        job_id = filename.split("_")[0]
        code_distance, dt, shots, initial_logical_state = map(int, filename.split("_")[1:])

        return {
            "file_name": filename,
            "job_id": job_id,
            "code_distance": code_distance,
            "ancillas": code_distance - 1,
            "dt": dt,
            "shots": shots,
            "initial_logical_state": initial_logical_state,
            "no_non_trivial_shots": None
        }
            
    def generate_jobdata(self):
        raise NotImplementedError
    
    def load_jobdata(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Parse one JSON job, extract detection events, build a complete
        Chebyshev-distance graph for each nontrivial shot:
    
        Keyword arguments:
        job -- path to the job file
        jon_params -- job parameters (as dict) for the job file
        Returns:
        data -- pythorch dataset with graphs and correct labels with correct format for GNN-network
        """
        job = self.job_dir / self.filename
        result = None
        with open(job) as f:
            result = json.load(f,cls=RuntimeDecoder)[0]

        ### EXTRACT SYNDROME
        def get_syndrome_from_qbits(qbits: list[str]) -> np.ndarray:
            arr = np.array([[int(c) for c in s] for s in qbits], dtype=np.uint8)
            return arr[:, :-1] ^ arr[:, 1:]
        
        def get_final_logical_state(final_state: np.ndarray) -> np.ndarray:
            # Count the number of '1's in final_state and take modulo 2, initial state parity will always be zero
            diff_parity = np.array([s.count("1") % 2 == 1 for s in final_state])
            matrix = np.full((len(diff_parity), self.t), False, dtype=bool) # Match dimensions of matrix from logical readings at all time steps
            matrix[:, -1] = diff_parity
            return matrix
        
        def extract_flip_matrix(shot_list: list[str], ancillas: int) -> np.ndarray:
            bit_array = np.array([[int(bit) for bit in shot] for shot in shot_list], dtype=np.uint8)
            T = bit_array.shape[1] // ancillas
            mat = bit_array.reshape(-1, T, ancillas)
            flips = np.diff(mat, axis=1).astype(bool)
            return flips.reshape(flips.shape[0], -1)
        
        syndrome_excluding_initial_final = result.data.syndrome.get_bitstrings()
        syndrome_excluding_initial_final = [s[::-1] for s in syndrome_excluding_initial_final] #Flip IBM data
        
        final_state = result.data.final_state.get_bitstrings()
        final_state = [s[::-1] for s in final_state] #Flip IBM data


        initial_syndrome = np.array([[int(bit) for bit in str(self.job_params["initial_logical_state"])*self.job_params["ancillas"]]] * self.job_params["shots"], dtype=np.uint8)
        syndrome_excluding_initial_final = np.array([[int(c) for c in s] for s in syndrome_excluding_initial_final], dtype=np.uint8)
        final_syndrome = get_syndrome_from_qbits(final_state)

        syndrome = np.concatenate([initial_syndrome, syndrome_excluding_initial_final, final_syndrome], axis=1)
        
        detector_events_list = extract_flip_matrix(syndrome, self.job_params["ancillas"]) # CREATE NODES/DETECTION EVENTS
        flips_array = get_final_logical_state(final_state)

        return detector_events_list, flips_array


if __name__ == "__main__":
    sampler = IBM_sampler(distance=3, t=5)
    detection_events, observable_flips = sampler.load_jobdata()


