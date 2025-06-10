import numpy as np
import torch
from torch_geometric.data import Data
import os
import json
from args import Args
from qiskit_ibm_runtime import RuntimeDecoder
import sys
import networkx as nx
import matplotlib.pyplot as plt
from utils import parse_yaml, get_job_params
from concurrent.futures import ProcessPoolExecutor, as_completed

class IBM_sampler:
    """Class for creating graphs and correct labels for a GNN-network from job files,
    the graphs and correct labels are saved to a pythorh file for each job file."""
    
    def __init__(self, args: Args):

        paths, model_settings, graph_settings, training_settings = parse_yaml(yaml_config)
        self.get_job_dir = paths["get_job_dir"] 
        self.graph_dir = paths["graph_dir"]    
    
    def parse_json(self):
        job_dir = "ibm_jobdata"

    def create_graph(self, job, job_params):
        """
        Parse one JSON job, extract detection events, build a complete
        Chebyshev-distance graph for each nontrivial shot:
    
        Keyword arguments:
        job -- path to the job file
        jon_params -- job parameters (as dict) for the job file
        Returns:
        data -- pythorch dataset with graphs and correct labels with correct format for GNN-network
        """
        result = None
        try:
            with open(job) as f:
                result = json.load(f,cls=RuntimeDecoder)[0]
        except:
            with open(job) as f:
                result = json.load(f,cls=RuntimeDecoder)


        ### EXTRACT SYNDROME
        def get_syndrome_from_qbits(qbits):
            # qbits is list of bit-strings or lists like [['0','1',…],…]
            arr = np.array([list(s) for s in qbits], dtype='<U1').astype(np.int8)
            # XOR adjacent columns
            diffs = arr[:, :-1] ^ arr[:, 1:]       # shape (shots, d-1)
            # Join each row’s bits with minimal overhead
            return [''.join(map(str, row)) for row in diffs]
        
        initial_symdrome = [(str(job_params["initial_logical_state"])*job_params["ancillas"])]*job_params["shots"]
        
        syndrome_excluding_initial_final = result.data.syndrome.get_bitstrings()
        syndrome_excluding_initial_final = [s[::-1] for s in syndrome_excluding_initial_final] #Flip IBM data
        
        final_state = result.data.final_state.get_bitstrings()
        final_symdrome = get_syndrome_from_qbits(final_state)

        syndrome = [initial_symdrome[i] + syndrome_excluding_initial_final[i] + final_symdrome[i] for i in range(job_params["shots"])]

def get_job_params(filename):
    """Get the job parameters from a job file name

    The expected filename format is:
        <job_id>_<code_distance>_<d_t>_<shots>_<initial_logical_state>.<extension>

    For example:
        "job42_7_5_1000_1.json" would give:
            job_id = "job42"
            code_distance = 7
            d_t = 5
            shots = 1000
            initial_logical_state = 1
    """
    filename = filename.split(".")[0] #Remove file ending
    filename = filename.split("/")[-1] #Remove path
    job_id = filename.split("_")[0]
    code_distance, d_t, shots, initial_logical_state = map(int,filename.split("_")[1:])
    
    return {
        "file_name": filename,
        "job_id": job_id,
        "code_distance": code_distance,
        "ancillas": code_distance-1,
        "d_t": d_t,
        "shots": shots,
        "initial_logical_state": initial_logical_state,
        "no_non_trivial_shots": None
        }

if __name__ == "__main__":
    # Directory containing your job files
    job_dir = "jobs/training_data"

    # Gather all files (adjust extension filter as needed)
    job_files = ["training_data/"+f.split(".")[0] for f in os.listdir(job_dir) ]


