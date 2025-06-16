import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from repetition_code.repetition_code import QuantumErrorCorrection


def run_qec_job(code_distance, time_steps):
    qec = QuantumErrorCorrection(code_distance=code_distance, time_steps=time_steps, shots=20000, initial_state=0, simulator=True)
    #return qec.execute_batch(repetitions=2)
    return qec.execute()

code_distances = [3]
time_steps_list = [13]

# Lista med förbjudna kombinationer
excluded_combinations = {} #{(5, 5), (3,5),(3,3), (9,3), (7,7), (5,9), (9,5), (9, 9), (13,13), (9,13)}  # Lägg till fler om du vill

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