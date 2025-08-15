import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir)) # Add parent directory to Python path

import json

from qiskit import transpile, QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder, SamplerV2 as Sampler
from qiskit_aer import AerSimulator

from repetition_code.repetition_code_circuit import RepetitionCodeCircuit

class RepetitionCodeExecute:
    """
    Build and execute quantum repetition code circuits using Qiskit,
    either on real IBM Quantum backends or in simulation.
    """

    def __init__(
        self,
        code_distance: int,
        time_steps: int,
        shots: int,
        initial_state: int,
        simulator: bool,
        noise_angle: float,
        backend_name: str,
        meas_type: str = "classified",
        subdir: str = "",
    ):
        """
        Initialize repetition code parameters and connect to a backend.

        :param code_distance: Repetition code distance (must be an odd integer: 3, 5, 7, ...)
        :param time_steps: Number of syndrome measurement rounds
        :param shots: Number of repetitions for circuit execution
        :param initial_state: Initial logical state (0 or 1)
        :param simulator: If True, run on Aer simulator; if False, run on real backend
        :param noise_angle: Noise rotation angle in radians
        :param backend_name: Name of IBM Quantum backend to use
        :param meas_type: Measurement type for hardware execution ("kerneled" for IQ data, "classified" for binary), IQ data does not work on Aer simulator
        :param subdir: Optional subdirectory name for saving results
        """
        self.code_distance = code_distance
        self.num_qubits = 2 * code_distance - 1
        self.time_steps = time_steps
        self.shots = shots
        self.initial_state = initial_state
        self.simulator = simulator
        self.noise_angle = noise_angle
        self.backend_name = backend_name
        self.meas_type = meas_type
        self.subdir = subdir

        # Connect to backend
        self.service = QiskitRuntimeService()
        self.backend = self.service.backend(self.backend_name)

        print(
            f"Connected to: {self.backend.name}, "
            f"code distance: {self.code_distance}, "
            f"time steps: {self.time_steps}, "
            f"measurement type: {self.meas_type}"
        )

    def transpile_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Transpile the circuit for the target backend without routing swaps.

        :param circuit: Quantum circuit to optimize
        :return: Transpiled circuit
        """
        layout = None  # Set to fixed physical layout for reproducibility, if desired
        # layout = [1, 3, 5, 7, 27, 29, 49, 51, 71, 73, 93, 95, 115, 113, 0, 2, 4, 6, 17, 28, 38, 50, 58, 72, 79, 94, 99, 114, 119]

        transpiled = transpile(
            circuit,
            backend=self.backend,
            initial_layout=layout,
            routing_method="none",  # Disable SWAP-based routing
            optimization_level=1,
            seed_transpiler=42,
        )

        print("Physical layout:", transpiled.layout.final_index_layout(filter_ancillas=True))
        return transpiled

    def execute(self):
        """
        Run the repetition code circuit on the chosen backend
        and save the results to a JSON file.
        """
        # Build repetition code circuit
        code = RepetitionCodeCircuit(
            self.code_distance,
            self.time_steps,
            resets=False,
            xbasis=True,
            barriers=True,
            noise_angle=self.noise_angle,
        )
        circuit = code.circuit[str(self.initial_state)]
        transpiled_circuit = self.transpile_circuit(circuit)

        if self.simulator:
            # Run on Aer simulator
            simulator = AerSimulator.from_backend(self.backend)
            job = simulator.run(transpiled_circuit, shots=self.shots, seed_simulator=42)
            result = job.result()

            filename = (
                f"./jobdata/aer/"
                f"{job.job_id()}_{self.code_distance}_{self.time_steps}_"
                f"{self.shots}_{self.initial_state}.json"
            )
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as file:
                json.dump(result, file, cls=RuntimeEncoder)

            time_taken = result.to_dict()["time_taken"]
            print(
                f"Measurement saved as '{filename}', "
                f"simulated sampling took {time_taken:.1f} s."
            )

        else:
            # Run on real IBM backend with chosen measurement type
            sampler = Sampler(
                mode=self.backend,
                options={
                    "default_shots": self.shots,
                    "execution": {"meas_type": self.meas_type},
                },
            )
            job = sampler.run([transpiled_circuit])
            result = job.result()

            res_filename = os.path.join(
                f"./jobdata/ibm{self.subdir}/",
                f"{job.job_id()}_{self.code_distance}_{self.time_steps}_"
                f"{self.shots}_{self.initial_state}_{self.noise_angle:.4f}.json",
            )
            os.makedirs(os.path.dirname(res_filename), exist_ok=True)
            with open(res_filename, "w") as file:
                json.dump(result, file, cls=RuntimeEncoder)

            print(f"Measurement saved as '{res_filename}'")
            return result


if __name__ == "__main__":
    qec = RepetitionCodeExecute(
        code_distance=15,
        time_steps=49,
        shots=1_000_000,
        initial_state=0,
        simulator=False,
        noise_angle=0.1653,
        backend_name="ibm_pittsburgh",
        meas_type="classified",  # Change to "classified" for binary data, kerneled for IQ-data
        subdir="/turning_the_knob",
    )
    qec.execute()