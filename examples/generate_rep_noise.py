import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from repetition_code.repetition_code_execute import RepetitionCodeExecute  # justera importen efter din struktur

def run_experiment(alpha):
    qec = RepetitionCodeExecute(
        code_distance=49,
        time_steps=49,
        shots=20_000,
        initial_state=0,
        simulator=False,
        noise_angle=alpha,
        subdir="/noise_angle_pittsburgh"
    )
    qec.execute()
    return alpha


if __name__ == "__main__":
    # Definiera alfa-intervall
    alphas = np.linspace(0, np.pi/2, 20, endpoint=True)  # 0, π/50, …, 4π/50
    
    # Välj antal trådar (t.ex. lika med antal alphas eller cpu-kärnor)
    max_workers = len(alphas)
    
    # Skapa pool och starta alla experiment parallellt
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Skicka in alla uppgifter
        futures = {executor.submit(run_experiment, a): a for a in alphas}
        
        # Hämta och hantera resultaten när de blir klara
        for future in as_completed(futures):
            alpha = future.result()
            print(f"Alpha = {alpha:.3f}")