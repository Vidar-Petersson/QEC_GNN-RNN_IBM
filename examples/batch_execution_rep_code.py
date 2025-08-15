import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))  # Add parent dir for imports

import numpy as np
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from repetition_code.repetition_code_execute import RepetitionCodeExecute 


def run_qec_job(code_distance: int, time_steps: int, noise_angle: float):
    """
    Run a single quantum error correction experiment on real hardware.

    Args:
        code_distance (int): The code distance of the repetition code (must be odd).
        time_steps (int): Number of syndrome measurement rounds.
        noise_angle (float): Noise injection angle in radians.
    """
    qec = RepetitionCodeExecute(
        code_distance=code_distance,
        time_steps=time_steps,
        shots=100_000,                # Number of shots per run
        initial_state=0,               # Start in logical |0>
        simulator=False,               # Run on IBM-hardware
        noise_angle=noise_angle,       # Injected noise rotation
        subdir="/noise_angle"    # Output directory
    )
    qec.execute()
    return code_distance, time_steps, noise_angle


if __name__ == "__main__":
    # Parameter sweeps
    code_distances = [49]
    time_steps_list = [49] 
    noise_angles = np.linspace(0, np.pi / 2, 20, endpoint=True)

    # Excluded combinations (optional)
    excluded_combinations = set()  # e.g., {(5, 5, 0.0)}

    # Create full parameter set excluding forbidden ones
    parameter_combinations = [
        (d, t, a)
        for d, t, a in product(code_distances, time_steps_list, noise_angles)
        if (d, t, a) not in excluded_combinations
    ]

    # Parallel execution
    with ThreadPoolExecutor(max_workers=len(parameter_combinations)) as executor:
        futures = {
            executor.submit(run_qec_job, d, t, a): (d, t, a)
            for d, t, a in parameter_combinations
        }

        for future in as_completed(futures):
            d, t, a = futures[future]
            try:
                future.result()
                print(f"Completed: code_distance={d}, time_steps={t}, noise_angle={a:.4f} rad")
            except Exception as e:
                print(f"Error: code_distance={d}, time_steps={t}, noise_angle={a:.4f} rad: {e}")
