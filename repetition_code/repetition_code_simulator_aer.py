import json
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
from qiskit_ibm_runtime import RuntimeEncoder

class QuantumErrorCorrectionSim:
    """
    Klass för att bygga och köra simulerade kvantfelkorrigeringskretsar med Qiskit Aer.
    """
    def __init__(self, code_distance: str, time_steps: str, shots: str, initial_state: int = 0):
        """
        Initierar systemets parametrar och simulatorn.
        
        :param code_distance: Kodavstånd (måste vara ett udda heltal: 3, 5, 7, ...)
        :param time_steps: Antal syndrommätningar
        :param shots: Antal exekveringar av kretsen
        """
        self.code_distance = code_distance
        self.num_qubits = 2 * code_distance - 1
        self.time_steps = time_steps
        self.shots = shots
        self.initial_state = initial_state
        
        self.device_backend = FakeMarrakesh()
        self.simulator = AerSimulator.from_backend(self.device_backend)
        
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
            circuit.cx(self.state_data, redundance)
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
            with circuit.if_test((self.creg_syndrome[offset + i], 1)):
                circuit.x(self.qreg_ancillas[i])
        
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

    def execute(self) -> object:
        """ Kör kvantkretsen på Qiskit Aer backend och sparar resultatet. """
        circuit = self.build_error_correction_sequence()

        # Kör på Qiskit Aer sampler
        simulator = self.simulator
        transpiled_circuit = transpile(circuit, simulator)
        job = simulator.run([transpiled_circuit], shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        syndromes = []
        final_states = []
        for count in counts.keys():
            split_count = count.split()
            syndrome = split_count[0]
            final_state = split_count[1]
            for i in range(counts[count]):
                syndromes.append(syndrome)
                final_states.append(final_state)
        return result, syndromes, final_states

if __name__ == "__main__":
    qec = QuantumErrorCorrectionSim(code_distance=5, time_steps=12, shots=32000, initial_state=0)
    result, syndromes, final_states = qec.execute()
    
    result_dict = result.to_dict()
    result_dict['results'][0]['data']['syndrome'] = syndromes
    result_dict['results'][0]['data']['final_state'] = final_states
    
    filename = f"./jobs/simulations/aer_{qec.code_distance}_{qec.time_steps}_{qec.shots}_{qec.initial_state}.json"
    with open(filename, "w") as file:
        json.dump(result_dict, file, cls=RuntimeEncoder)
    
    print(f"Output sparad som '{filename}'.")