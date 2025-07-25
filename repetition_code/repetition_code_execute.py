import json
import numpy as np

from qiskit import transpile, QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder, SamplerV2 as Sampler
from qiskit_aer import AerSimulator 

from repetition_code.repetition_code_circuit import RepetitionCodeCircuit

class RepetitionCodeExecute:
    """
    Class for building and running quantum error correction circuits with Qiskit, both experimentally and in simulation.
    """
    def __init__(self, code_distance: int, time_steps: int, shots: int, initial_state: int, simulator: bool, noise_angle: float):
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

        self.service = QiskitRuntimeService()
        self.backend = self.service.backend("ibm_kingston")  # Specify backend
        print("Connected to:", self.backend.name, "with distance:", self.code_distance, ", repetitions:", self.time_steps)
        
    def optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        # from qiskit_qec.qubit_selector.backend_evaluator import BackendEvaluator

        # evaluator = BackendEvaluator(self.backend)
        # print("Evaluating path...")
        # path, fidelity, num_subsets = evaluator.evaluate(self.num_qubits)
        # link_qubits = path[1::2]
        # code_qubits = path[0::2]
        # layout = link_qubits + code_qubits

        layout = [27, 29, 31, 11, 13, 15, 35, 33, 53, 51, 71, 73, 93, 91, 89, 87, 107, 109, 129, 127, 125, 105, 103, 101, 17, 28, 30, 18, 12, 14, 19, 34, 39, 52, 58, 72, 79, 92, 90, 88, 97, 108, 118, 128, 126, 117, 104, 102, 116]
        #layout = [0,2,4,6,8,10,12,14,19,34,32,30,38,48,57,68,70,72,79,92,90,88,86,84,96, 1,3,5,7,9,11,13,15,35,33,31,29,49,47,67,69,71,73,93,91,89,87,85,83]

        transpiled = transpile(circuit, backend=self.backend,
                            #initial_layout=layout,
                            #routing_method='none',
                            #layout_method='trivial',
                            optimization_level=1,
                            seed_transpiler=42)
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
            # Run on a real IBM backend
            sampler = Sampler(self.backend)
            job = sampler.run([transpiled_circuit], shots=self.shots)
            result = job.result()
            filename = f"./jobdata/ibm/noise_angle/{job.job_id()}_{self.code_distance}_{self.time_steps}_{self.shots}_{self.initial_state}_{self.noise_angle:.4f}.json"

            with open(filename, "w") as file:
                json.dump(result, file, cls=RuntimeEncoder)
            
            print(f"Measurement saved as '{filename}'.")
        return result

if __name__ == "__main__":
    alphas = np.linspace(0, 4/50, 11)  # 0, π/50, 2π/50, …, 4π/50
    for alpha in alphas:
        qec = RepetitionCodeExecute(code_distance=49, time_steps=49, shots=20_000, initial_state=0, simulator=False, noise_angle=alpha)
        qec.execute()