import json
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder, SamplerV2 as Sampler, Batch
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

class QuantumErrorCorrection:
    """
    Klass för att bygga och köra kvantfelkorrigeringskretsar med Qiskit.
    """
    def __init__(self, code_distance: str, time_steps: str, shots: str, initial_state: int = 0):
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

        self.service = QiskitRuntimeService()
        #self.backend = self.service.least_busy(operational=True, simulator=False, min_num_qubits=self.num_qubits)
        self.backend = self.service.backend("ibm_marrakesh")
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

        # Exempel: välj fysiska kvantbitar (måste passa maskinen!)
        layout = {self.qreg_data[i]: i+6 for i in range(self.code_distance)}
        ancilla_offset = self.code_distance
        for i in range(self.num_qubits - self.code_distance):
            layout[self.qreg_ancillas[i]] = ancilla_offset + i +6

        transpiled = transpile(circuit, backend=self.backend,
                            #initial_layout=layout,
                            optimization_level=2,
                            seed_transpiler=42)
        return transpiled

        """ Optimerar kretsen för att minska antalet grindar. """
        transpiled = transpile(circuit, backend=self.backend, optimization_level=2, seed_transpiler=42)
        return transpiled
    
    def execute(self) -> object:
        """ Kör kvantkretsen på backend och sparar resultatet. """
        circuit = self.build_error_correction_sequence()
        optimized_circuit = self.optimize_circuit(circuit)

        # Kör på riktig IBM-backend
        sampler = Sampler(self.backend)
        job = sampler.run([optimized_circuit], shots=self.shots)
        result = job.result()
        filename = f"./jobs/small_jobs/{job.job_id()}_{self.code_distance}_{self.time_steps}_{self.shots}_{self.initial_state}.json"

        with open(filename, "w") as file:
            json.dump(result, file, cls=RuntimeEncoder)
        
        print(f"Output sparad som '{filename}'.")
        return result

    def execute_batch(self, repetitions: int = 5) -> list:
        """
        Kör flera identiska jobb i en batch och returnerar alla resultat.
        
        :param repetitions: Antal upprepningar av körningen
        :return: Lista med resultatobjekt
        """
        circuit = self.build_error_correction_sequence()
        optimized_circuit = self.optimize_circuit(circuit)

        results = []
        backend = self.backend

        with Batch(backend=backend) as batch:
            sampler = Sampler(mode=batch)
            jobs = []

            for i in range(repetitions):
                job = sampler.run([optimized_circuit], shots=self.shots)
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

def run_qec_job(code_distance, time_steps):
    qec = QuantumErrorCorrection(code_distance=code_distance, time_steps=time_steps, shots=20000, initial_state=0)
    #return qec.execute_batch(repetitions=2)
    return qec.execute()

if __name__ == "__main__":
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