import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from repetition_code.repetition_code_old import QuantumErrorCorrection
import numpy as np


def run_qec_job(code_distance, time_steps):
    qec = QuantumErrorCorrection(code_distance=code_distance, time_steps=time_steps, shots=20_000, initial_state=0, simulator=False, angle_scale=np.pi/25)
    #return qec.execute_batch(repetitions=2)
    return qec.execute()

# code_distances = [3,5,7,9,11,13,15,17,19,21,23,25]
# code_distances = [3,7,11,15,19,23]
code_distances = [11]
# time_steps_list = [9, 24, 49, 74, 99, 249, 499, 749, 999]
time_steps_list = [3]

# Lista med förbjudna kombinationer
excluded_combinations = {} 

# Skapa alla tillåtna kombinationer
parameter_combinations = [
    (d, t) for d, t in product(code_distances, time_steps_list)
    if (d, t) not in excluded_combinations
]

with ThreadPoolExecutor(max_workers=len(parameter_combinations)) as executor:
    futures = {
        executor.submit(run_qec_job, d, t): (d, t) for d, t in parameter_combinations
    }
    for future in as_completed(futures):
        d, t = futures[future]
        try:
            result = future.result()
            print(f"Klar: code_distance={d}, time_steps={t}")
        except Exception as e:
            print(f"Fel: code_distance={d}, time_steps={t}: {e}")