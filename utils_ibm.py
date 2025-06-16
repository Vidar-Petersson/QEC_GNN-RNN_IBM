import numpy as np
from pathlib import Path
import os
import re
import json
from qiskit_ibm_runtime import RuntimeDecoder

class IBM_sampler:
    """Class for loading detection_events and observable_flips from existing json data.
    If jobdata doesn't exists for the desired config, ask to generate it via IBM Quantum."""
    
    def __init__(self, distance: int, t: int, simulated=False):
        self.simulated = simulated
        self.distance = distance 
        self.t = t
        self.job_dir, self.filename = self._find_filename(self.simulated, self.distance, self.t-1)
        self.job_params = self._get_job_params(self.filename)
        self.device = "cpu"


    @staticmethod
    def _find_filename(simulated: bool, d: int, t: int) -> str:
        """Find the filename in the given directory that matches the pattern _<d>_<t>_"""
        if simulated:
            job_dir = Path("./jobdata/aer/")
        else:
            job_dir = Path("./jobdata/ibm")

        pattern = re.compile(rf"_({d})_({t})_")
        for filename in os.listdir(job_dir):
            if pattern.search(filename):
                return job_dir, filename
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

        ### EXTRACT SYNDROME
        def get_syndrome_from_qbits(qbits: list[str]) -> np.ndarray:
            arr = np.array([[int(c) for c in s] for s in qbits], dtype=np.uint8)
            return arr[:, :-1] ^ arr[:, 1:]
        
        def get_final_logical_state(final_state: np.ndarray) -> np.ndarray:
            diff_parity = np.array([s[0] == "1" for s in final_state]) # 0/1 klassning
            # diff_parity = np.array([s.count("1") % 2 == 1 for s in final_state]) # Jämn/udda klassning 
            matrix = np.full((len(diff_parity), self.t), False, dtype=bool) # Match dimensions of matrix from logical readings at all time steps
            matrix[:, -1] = diff_parity
            return matrix
        
        def extract_flip_matrix(shot_list: list[str], ancillas: int) -> np.ndarray:
            bit_array = np.array([[int(bit) for bit in shot] for shot in shot_list], dtype=np.uint8)
            T = bit_array.shape[1] // ancillas
            mat = bit_array.reshape(-1, T, ancillas)
            flips = np.diff(mat, axis=1).astype(bool)
            return flips.reshape(flips.shape[0], -1)
        
        def extract_jobdata(simulated: bool) -> tuple[list, list]:
            job = self.job_dir / self.filename

            if simulated:
                with open(job) as f:
                    result = json.load(f,cls=RuntimeDecoder)
                    counts = result.get_counts()
                    syndrome_excluding_initial_final = []
                    final_state = []
                    for bitstring, freq in counts.items():
                        syndromes, final_states = bitstring.split()
                        syndrome_excluding_initial_final.extend([syndromes] * freq)
                        final_state.extend([final_states] * freq)
            else:
                result = None
                with open(job) as f:
                    result = json.load(f,cls=RuntimeDecoder)[0]
                syndrome_excluding_initial_final = result.data.syndrome.get_bitstrings()
                final_state = result.data.final_state.get_bitstrings()

            return syndrome_excluding_initial_final, final_state
        
        syndrome_excluding_initial_final, final_state = extract_jobdata(self.simulated)
        syndrome_excluding_initial_final = [s[::-1] for s in syndrome_excluding_initial_final] #Flip IBM data
        final_state = [s[::-1] for s in final_state] #Flip IBM data

        initial_syndrome = np.array([[int(bit) for bit in str(self.job_params["initial_logical_state"])*self.job_params["ancillas"]]] * self.job_params["shots"], dtype=np.uint8)
        syndrome_excluding_initial_final = np.array([[int(c) for c in s] for s in syndrome_excluding_initial_final], dtype=np.uint8)
        final_syndrome = get_syndrome_from_qbits(final_state)

        syndrome = np.concatenate([initial_syndrome, syndrome_excluding_initial_final, final_syndrome], axis=1)
        
        detector_events_list = extract_flip_matrix(syndrome, self.job_params["ancillas"]) # CREATE NODES/DETECTION EVENTS
        flips_array = get_final_logical_state(final_state)

        return detector_events_list, flips_array

if __name__ == "__main__":
    sampler = IBM_sampler(distance=3, t=6, simulated=False)
    detection_events, observable_flips = sampler.load_jobdata()