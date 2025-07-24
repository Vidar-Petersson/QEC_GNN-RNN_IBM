import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from qiskit_qec.circuits.repetition_code import RepetitionCodeCircuit

import json


from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import IfElseOp
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder, SamplerV2 as Sampler, Batch
from qiskit_aer import AerSimulator 

class QuantumErrorCorrection:
    """
    Class for building and running quantum error correction circuits with Qiskit, both experimentally and in simulation.
    """
    def __init__(self, code_distance: int, time_steps: int, shots: int, initial_state: int, simulator: bool, angle_scale: float):
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
        self.angle_scale = angle_scale

        self.service = QiskitRuntimeService()
        self.backend = self.service.backend("ibm_kingston")  # Specify backend
        #self.backend.target.add_instruction(IfElseOp, name="if_else")
        print("Connected to:", self.backend.name, "with distance:", self.code_distance, ", repetitions:", self.time_steps)
        
    
    def optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        # # Example: choose physical qubit layout (must match the hardware!)
        # layout = {self.qreg_data[i]: i for i in range(self.code_distance)}
        
        # for i in range(self.num_qubits - self.code_distance):
        #     layout[self.qreg_ancillas[i]] = self.code_distance + i

        # Find layout
        # from qiskit_qec.qubit_selector.backend_evaluator import BackendEvaluator

        # evaluator = BackendEvaluator(self.backend)
        # print("Evaluating path...")
        # path, fidelity, num_subsets = evaluator.evaluate(self.num_qubits)
        # print(path)
        # link_qubits = path[1::2]
        # code_qubits = path[0::2]
        # layout = link_qubits + code_qubits
        # print(layout)
        #asdasd
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
        code = RepetitionCodeCircuit(self.code_distance, self.time_steps, resets=False, xbasis=True, barriers=True)
        circuit = code.circuit["0"]
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
            #sampler.options.experimental = {"execution_path" : "gen3-experimental"}
            job = sampler.run([transpiled_circuit], shots=self.shots)
            result = job.result()
            filename = f"./jobdata/ibm/{job.job_id()}_{self.code_distance}_{self.time_steps}_{self.shots}_{self.angle_scale:.4f}.json"

            with open(filename, "w") as file:
                json.dump(result, file, cls=RuntimeEncoder)
            
            print(f"Measurement saved as '{filename}'.")
        return result

if __name__ == "__main__":
    qec = QuantumErrorCorrection(code_distance=25, time_steps=49, shots=20_000, initial_state=0, simulator=False, angle_scale=0)
    qec.execute()