import json
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder, SamplerV2 as Sampler, Batch
from qiskit_aer import AerSimulator

class QuantumErrorCorrection:
    """
    Klass för att bygga och köra kvantfelkorrigeringskretsar med Qiskit, både experimentellt och simulerat.
    """
    def __init__(self, code_distance: int, time_steps: int, shots: int, initial_state: int, simulator: bool):
        """
        Initierar systemets parametrar och ansluter till en backend.
        
        :param code_distance: Kodavstånd (måste vara ett udda heltal: 3, 5, 7, ...)
        :param time_steps: Antal syndrommätningar
        :param shots: Antal exekveringar av kretsen
        """
        self.code_distance = code_distance
        self.num_qubits = 2 * code_distance - 1
        self.time_steps = time_steps
        self.shots = shots
        self.initial_state = initial_state
        self.simulator = simulator

        self.service = QiskitRuntimeService()
        self.backend = self.service.backend("ibm_marrakesh") # Specify backend
        print("Connected to:", self.backend, "with distance:", self.code_distance, ", time: ", self.time_steps)
        
        # Registrerar kvant- och klassiska bitar
        self.qreg_data = QuantumRegister(self.code_distance)  # Dataqubits
        self.qreg_ancillas = QuantumRegister(self.num_qubits - self.code_distance)  # Mätqubits
        self.creg_syndrome = ClassicalRegister(self.time_steps * (self.code_distance - 1), name="syndrome") # Klassiska bitar för syndromdata
        self.creg_final_state = ClassicalRegister(self.code_distance, name="final_state")  # Klassiska bitar för mätdata
        
        self.state_data = self.qreg_data[0]  # Initialtillstånd
        self.redundances_data = self.qreg_data[1:]  # Redundansqubits
    
    def build_qc(self) -> QuantumCircuit:
        """ Skapar en kvantkrets med registrerade qubits. """
        return QuantumCircuit(self.qreg_data, self.qreg_ancillas, self.creg_final_state, self.creg_syndrome)
    
    def initialize_qubits(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """ Initialiserar qubits i ett likformigt superpositionstillstånd och sammanflätar redundanta qubits. """
        if self.initial_state == 1:
            circuit.x(self.qreg_data)
        circuit.h(self.qreg_data)
        circuit.barrier(self.qreg_data)

        for redundance in self.redundances_data:
            circuit.cx(redundance, self.state_data)
        circuit.barrier(self.qreg_data, *self.qreg_ancillas)
        return circuit
    
    def measure_syndrome_bit(self, circuit: QuantumCircuit, offset: int) -> QuantumCircuit:
        """
        Mäter syndrombitar genom att beräkna paritet för intilliggande qubits och lagra resultaten i klassiska bitar.
        
        :param circuit: Kvantkretsen som modifieras
        :param offset: Offset i de klassiska registren för att spara syndromvärdena
        """
        circuit.h(self.qreg_ancillas) # Initialisera alla i |+> / |->
        circuit.barrier(self.qreg_data, *self.qreg_ancillas)

        for i in range(self.code_distance - 1):
            circuit.cx(self.qreg_ancillas[i], self.qreg_data[i])
            circuit.cx(self.qreg_ancillas[i], self.qreg_data[i + 1])
        circuit.barrier(*self.qreg_data, *self.qreg_ancillas)

        circuit.h(self.qreg_ancillas)
        circuit.barrier(*self.qreg_data, *self.qreg_ancillas)

        # Mätning av syndrombitar
        for i in range(self.code_distance - 1):
            circuit.measure(self.qreg_ancillas[i], self.creg_syndrome[offset + i])
        circuit.barrier(*self.qreg_data, *self.qreg_ancillas)
        
        # Reset av mätqubits för återanvändning
        for i in range(self.code_distance - 1):
            circuit.reset(self.qreg_ancillas[i])
        
        circuit.barrier(*self.qreg_data, *self.qreg_ancillas)
        return circuit
    
    def apply_final_readout(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """ Mäter och sparar slutliga värden på dataqubits. """
        circuit.barrier(self.qreg_data)
        circuit.h(self.qreg_data)
        circuit.measure(self.qreg_data, self.creg_final_state)
        return circuit
    
    def build_error_correction_sequence(self) -> QuantumCircuit:
        """ Bygger kvantkretsen. """
        circuit = self.build_qc() # Skapar alla klassiska och kvantbitar
        circuit = self.initialize_qubits(circuit) # Initialiserar bitarna superposition+sammanflätning
        for i in range(self.time_steps): # Gör syndrommätningarna
            circuit = self.measure_syndrome_bit(circuit, offset=(self.code_distance - 1) * i)
        circuit = self.apply_final_readout(circuit) # Mät anchillabitarna
        return circuit
    
    def optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        # # Exempel: välj fysiska kvantbitar (måste passa maskinen!)
        # layout = {self.qreg_data[i]: i+6 for i in range(self.code_distance)}
        # ancilla_offset = self.code_distance
        # for i in range(self.num_qubits - self.code_distance):
        #     layout[self.qreg_ancillas[i]] = ancilla_offset + i +6

        # transpiled = transpile(circuit, backend=self.backend,
        #                     #initial_layout=layout,
        #                     optimization_level=2,
        #                     seed_transpiler=42)
        # return transpiled

        """ Optimerar kretsen för att minska antalet grindar. """
        transpiled = transpile(circuit, backend=self.backend, optimization_level=2, seed_transpiler=42)
        return transpiled
    
    def execute(self) -> object:
        """ Kör kvantkretsen på backend och sparar resultatet. """
        circuit = self.build_error_correction_sequence()
        transpiled_circuit = self.optimize_circuit(circuit)

        if self.simulator:
            simulator = AerSimulator.from_backend(self.backend)
            job = simulator.run(transpiled_circuit, shots=self.shots, seed_simulator=42)
            result = job.result()
            
            filename = f"./jobdata/aer/{job.job_id()}_{self.code_distance}_{self.time_steps}_{self.shots}_{self.initial_state}.json"
            with open(filename, "w") as file:
                json.dump(result, file, cls=RuntimeEncoder)
            
            time = result.to_dict()["time_taken"]
            print(f"Mätning sparad som '{filename}', simulerad sampling tog {time:.1f} s.")
        else:
            # Kör på riktig IBM-backend
            sampler = Sampler(self.backend)
            job = sampler.run(transpiled_circuit, shots=self.shots)
            result = job.result()
            filename = f"./jobdata/ibm/{job.job_id()}_{self.code_distance}_{self.time_steps}_{self.shots}_{self.initial_state}.json"

            with open(filename, "w") as file:
                json.dump(result, file, cls=RuntimeEncoder)
            
            print(f"Mätning sparad som '{filename}'.")
        return result

    def execute_batch(self, repetitions: int = 5) -> list:
        """
        Kör flera identiska jobb i en batch och returnerar alla resultat.
        
        :param repetitions: Antal upprepningar av körningen
        :return: Lista med resultatobjekt
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
                jobs.append((i, job))  # Spara med index

            # Hämta resultat efteråt
            for i, job in jobs:
                result = job.result()
                filename = f"./jobs/training_data_same_qubit/{job.job_id()}_{self.code_distance}_{self.time_steps}_{self.shots}_{self.initial_state}.json"
                with open(filename, "w") as file:
                    json.dump(result, file, cls=RuntimeEncoder)
                print(f"Resultat {i} sparat som '{filename}'.")
                results.append(result)
        
        return results

if __name__ == "__main__":
    qec = QuantumErrorCorrection(code_distance=3, time_steps=100, shots=200000, initial_state=0, simulator=True)
    qec.execute()