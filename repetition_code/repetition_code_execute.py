import json
from datetime import datetime
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import numpy as np

from qiskit import transpile, QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder, SamplerV2 as Sampler
from qiskit_aer import AerSimulator 

from repetition_code.repetition_code_circuit import RepetitionCodeCircuit

class RepetitionCodeExecute:
    """
    Class for building and running quantum error correction circuits with Qiskit, both experimentally and in simulation.
    """
    def __init__(self, code_distance: int, time_steps: int, shots: int, initial_state: int, simulator: bool, noise_angle: float, subdir: str = ""):
        """
        Initializes the system parameters and connects to a backend.
        
        :param code_distance: Code distance (must be an odd integer: 3, 5, 7, ...)
        :param time_steps: Number of syndrome measurements
        :param shots: Number of circuit executions
        """
        self.code_distance = code_distance
        self.num_qubits = 2 * code_distance - 1
        self.time_steps = time_steps
        self.shots = shots
        self.initial_state = initial_state
        self.simulator = simulator
        self.noise_angle = noise_angle # Radians
        self.subdir = subdir

        self.service = QiskitRuntimeService()
        self.backend = self.service.backend("ibm_pittsburgh")  # Specify backend
        print("Connected to:", self.backend.name, "with distance:", self.code_distance, ", repetitions:", self.time_steps)
        
    def optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        # Find optimal path manually
        # from qiskit_qec.qubit_selector.backend_evaluator import BackendEvaluator
        # evaluator = BackendEvaluator(self.backend)
        # path, fidelity, num_subsets = evaluator.evaluate(self.num_qubits)
        # link_qubits = path[1::2]
        # code_qubits = path[0::2]
        # layout = link_qubits + code_qubits
        # print(layout)

        layout = [2, 4, 6, 8, 10, 12, 14, 19, 34, 32, 30, 28, 26, 24, 22, 36, 42, 44, 46, 48, 50, 52, 54, 59, 74, 72, 70, 68, 66, 64, 62, 76, 82, 84, 86, 88, 90, 92, 94, 99, 114, 112, 110, 108, 106, 104, 102, 116, 1, 3, 5, 7, 9, 11, 13, 15, 35, 33, 31, 29, 27, 25, 23, 21, 41, 43, 45, 47, 49, 51, 53, 55, 75, 73, 71, 69, 67, 65, 63, 61, 81, 83, 85, 87, 89, 91, 93, 95, 115, 113, 111, 109, 107, 105, 103, 101, 121]

        transpiled = transpile(circuit, backend=self.backend,
                            initial_layout=layout,
                            routing_method='none', # Disables routing via SWAP-operations
                            optimization_level=1,
                            seed_transpiler=42)
        

        print("Physical layout:", transpiled.layout.final_index_layout(filter_ancillas=True))
        
        return transpiled
    
    def execute(self) -> object:
        """ Runs the quantum circuit on the selected backend and saves the result. """
        code = RepetitionCodeCircuit(self.code_distance, self.time_steps, resets=False, xbasis=True, barriers=True, noise_angle=self.noise_angle)
        circuit = code.circuit[str(self.initial_state)]
        transpiled_circuit = self.optimize_circuit(circuit)

        if self.simulator:  # Use Aer simulator
            simulator = AerSimulator.from_backend(self.backend)
            job = simulator.run(transpiled_circuit, shots=self.shots, seed_simulator=42)
            result = job.result()
            
            filename = f"./jobdata/aer/{job.job_id()}_{self.code_distance}_{self.time_steps}_{self.shots}_{self.initial_state}.json"
            with open(filename, "w") as file:
                json.dump(result, file, cls=RuntimeEncoder)
            
            time = result.to_dict()["time_taken"]
            print(f"Measurement saved as '{filename}', simulated sampling took {time:.1f} s.")
        
        else:
            # --- Kör på riktigt IBM‑backend med IQ‑mätningar ---
            sampler = Sampler(
                mode=self.backend,
                options={
                    "default_shots": self.shots,
                    #"execution": {"meas_type": "kerneled"}
                }
            )
            job = sampler.run([transpiled_circuit])
            result = job.result()

            # 1) Spara hela result som tidigare
            res_filename = os.path.join(
                f"./jobdata/ibm{self.subdir}/",
                f"{job.job_id()}_{self.code_distance}_{self.time_steps}_{self.shots}_{self.initial_state}_{self.noise_angle:.4f}.json"
            )
            os.makedirs(os.path.dirname(res_filename), exist_ok=True)
            with open(res_filename, "w") as file:
                json.dump(result, file, cls=RuntimeEncoder)
            total_duration = 0.0
            # metadata = result.metadata
            # for span in metadata['execution']['execution_spans']:
            #     duration = (datetime.fromisoformat(span.stop) - datetime.fromisoformat(span.start)).total_seconds()
            #     total_duration += duration

            print(f"Measurement saved as '{res_filename}', experimental execution took {total_duration:.1f} s.")

            return result

if __name__ == "__main__":
    qec = RepetitionCodeExecute(code_distance=49, time_steps=50, shots=500, initial_state=0, simulator=False, noise_angle=0, subdir="/noise_angle_pittsburgh")
    qec.execute()