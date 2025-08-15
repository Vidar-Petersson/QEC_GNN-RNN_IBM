import json
import numpy as np 

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
        self.backend.target.add_instruction(IfElseOp, name="if_else")
        print("Connected to:", self.backend.name, "with distance:", self.code_distance, ", repetitions:", self.time_steps)
        
        # Register quantum and classical bits
        self.qreg_data = QuantumRegister(self.code_distance)  # Data qubits
        self.qreg_ancillas = QuantumRegister(self.num_qubits - self.code_distance)  # Measurement qubits
        self.creg_syndromes = ClassicalRegister(self.time_steps * (self.code_distance - 1), name="syndromes")  # Classical bits for syndrome data
        #self.creg_middle_states = ClassicalRegister(self.time_steps, name="middle_states")  # Classical bits for logical flag states
        self.creg_final_state = ClassicalRegister(self.code_distance, name="final_state")  # Classical bits for final measurements
        
        self.state_data = self.qreg_data[0]  # Initial logical state
        self.redundances_data = self.qreg_data[1:]  # Redundant qubits
    
    def build_qc(self) -> QuantumCircuit:
        """ Creates a quantum circuit with the registered qubits. """
        # return QuantumCircuit(self.qreg_data, self.qreg_ancillas, self.creg_syndromes, self.creg_middle_states, self.creg_final_state)
        return QuantumCircuit(self.qreg_data, self.qreg_ancillas, self.creg_syndromes, self.creg_final_state)
    
    def initialize_qubits(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """ Initializes qubits in a uniform superposition and entangles the redundant qubits. """
        if self.initial_state == 1:
            circuit.x(self.qreg_data)
        circuit.h(self.qreg_data)
        circuit.barrier(self.qreg_data)

        for redundance in self.redundances_data:
            circuit.cx(redundance, self.state_data)
        circuit.barrier(self.qreg_data, *self.qreg_ancillas)
        return circuit
    
    def measure_syndrome_bit(self, circuit: QuantumCircuit, time_repetition_idx: int) -> QuantumCircuit:
        """
        Measures syndrome bits by computing the parity of neighboring qubits and storing the result in classical bits.
        
        :param circuit: The quantum circuit being modified
        :param time_repetition_idx: Index for the time step (syndrome round)
        """
        circuit.h(self.qreg_ancillas)  # Initialize all ancillas in |+> / |-> states
        circuit.barrier(self.qreg_data, *self.qreg_ancillas)

        # Apply parity-check gates
        for i in range(self.code_distance - 1):
            circuit.cx(self.qreg_ancillas[i], self.qreg_data[i])
            circuit.cx(self.qreg_ancillas[i], self.qreg_data[i + 1])
        circuit.barrier(*self.qreg_data, *self.qreg_ancillas)

        circuit.h(self.qreg_ancillas)
        circuit.barrier(*self.qreg_data, *self.qreg_ancillas)

        offset = (self.code_distance - 1) * time_repetition_idx

        # Measure ancillas to get syndrome
        for i in range(self.code_distance - 1):
            circuit.measure(self.qreg_ancillas[i], self.creg_syndromes[offset + i])
        circuit.barrier(*self.qreg_data, *self.qreg_ancillas)

        # Optional: middle flag from data qubit 0
        # circuit.h(self.qreg_data[0])
        # circuit.measure(self.qreg_data[0], self.creg_middle_states[time_repetition_idx])
        # circuit.h(self.qreg_data[0])

        # Reset ancillas to |0>
        # for i in range(self.code_distance - 1):
        #     circuit.reset(self.qreg_ancillas[i])

        # Conditional X to reset ancillas to |0>
        for i in range(self.code_distance - 1):
            with circuit.if_test((self.creg_syndromes[offset + i], 1)):
                circuit.x(self.qreg_ancillas[i])

        if self.angle_scale > 0:
            self.inject_small_rotation_error(circuit, self.qreg_data, self.angle_scale)
        
        circuit.barrier(*self.qreg_data, *self.qreg_ancillas)
        return circuit
    
    def inject_small_rotation_error(self, circuit: QuantumCircuit, qubits: list, angle_scale=np.pi/50):
            """
            Injects a small random unitary rotation on a single qubit to simulate analog noise.
            
            :param circuit:
            :param qubit: The qubit (index or Qubit object) to apply the error to
            :param max_angle: Maximum rotation angle in radians (default: pi/50 ≈ 0.063)
            """
            # Draw a small rotation angle from a normal distribution centered at 0
            rng = np.random.default_rng(12345)
            theta = rng.normal(loc=0, scale=angle_scale)

            # Sample a random unit vector on the Bloch sphere (uniformly)
            phi = 2 * np.pi * np.random.rand()
            costheta = 2 * np.random.rand() - 1
            sintheta = np.sqrt(1 - costheta**2)
            nx = sintheta * np.cos(phi)
            ny = sintheta * np.sin(phi)
            nz = costheta

            # Apply a general unitary rotation about the (nx, ny, nz) axis with angle theta
            # Approximated using Qiskit's general single-qubit gate u(θ, φ, λ)
            circuit.u(theta * nx, theta * ny, theta * nz, qubits)
    
    def apply_final_readout(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """ Measures and stores the final values of the data qubits. """
        circuit.barrier(self.qreg_data)
        circuit.h(self.qreg_data)
        circuit.measure(self.qreg_data, self.creg_final_state)
        return circuit
    
    def build_error_correction_sequence(self) -> QuantumCircuit:
        """ Builds the full quantum error correction circuit. """
        circuit = self.build_qc()  # Create all classical and quantum bits
        circuit = self.initialize_qubits(circuit)  # Initialize in superposition and entangle
        for i in range(self.time_steps):  # Perform syndrome measurements
            circuit = self.measure_syndrome_bit(circuit, time_repetition_idx=i)
        circuit = self.apply_final_readout(circuit)  # Measure ancilla bits at the end
        return circuit
    
    def optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        # Example: choose physical qubit layout (must match the hardware!)
        layout = {self.qreg_data[i]: i for i in range(self.code_distance)}
        
        for i in range(self.num_qubits - self.code_distance):
            layout[self.qreg_ancillas[i]] = self.code_distance + i

        transpiled = transpile(circuit, backend=self.backend,
                            initial_layout=layout,
                            # routing_method='none',
                            layout_method='trivial',
                            optimization_level=1,
                            seed_transpiler=42)
        return transpiled
    
    def execute(self) -> object:
        """ Runs the quantum circuit on the selected backend and saves the result. """
        circuit = self.build_error_correction_sequence()
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
            sampler.options.experimental = {"execution_path" : "gen3-experimental"}
            job = sampler.run([transpiled_circuit], shots=self.shots)
            result = job.result()
            filename = f"./jobdata/ibm/{job.job_id()}_{self.code_distance}_{self.time_steps}_{self.shots}_{self.angle_scale:.4f}.json"

            with open(filename, "w") as file:
                json.dump(result, file, cls=RuntimeEncoder)
            
            print(f"Measurement saved as '{filename}'.")
        return result

    def execute_batch(self, repetitions: int = 5) -> list:
        """
        Runs multiple identical jobs in a batch and returns all results.
        
        :param repetitions: Number of repetitions of the run
        :return: List of result objects
        """
        circuit = self.build_error_correction_sequence()
        transpiled_circuit = self.optimize_circuit(circuit)

        results = []
        backend = self.backend

        with Batch(backend=backend) as batch:
            sampler = Sampler(mode=batch)
            jobs = []

            for i in range(repetitions):
                job = sampler.run(transpiled_circuit, shots=self.shots)
                jobs.append((i, job))  # Save with index

            # Retrieve results afterwards
            for i, job in jobs:
                result = job.result()
                filename = f"./jobs/training_data_same_qubit/{job.job_id()}_{self.code_distance}_{self.time_steps}_{self.shots}_{self.initial_state}_{self.angle_scale}.json"
                with open(filename, "w") as file:
                    json.dump(result, file, cls=RuntimeEncoder)
                print(f"Result {i} saved as '{filename}'.")
                results.append(result)
        
        return results

if __name__ == "__main__":
    qec = QuantumErrorCorrection(code_distance=3, time_steps=13, shots=20_000, initial_state=0, simulator=False, angle_scale=0)
    qec.execute()