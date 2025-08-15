import json
import stim

class QuantumErrorCorrectionStim:
    """
    Klass för att bygga och köra simulerade kvantfelkorrigeringskretsar med STIM.
    """
    def __init__(self, code_distance: str, time_steps: str, shots: str, initial_state: int = 0, error_rate: float = 0):
        """
        Initierar systemets parametrar.
        
        :param code_distance: Kodavstånd (måste vara ett udda heltal: 3, 5, 7, ...)
        :param time_steps: Antal syndrommätningar
        :param shots: Antal exekveringar av kretsen
        """
        self.code_distance = code_distance
        self.num_qubits = 2 * code_distance - 1
        self.time_steps = time_steps
        self.shots = shots
        self.initial_state = initial_state
        self.error_rate = error_rate
    
    def build_error_correction_sequence(self) -> stim.Circuit:
        """ Bygger kvantfelskorrigeringskretsen med komponenter från STIM samt lägger till depolariserings-errors inför operationer på qubits. """
        circuit = stim.Circuit()

        # Initialisera qubits
        qreg = [qb for qb in range(self.num_qubits)]
        data_qreg = [qb for qb in range(self.code_distance)]
        ancilla_qreg = [qb for qb in range(self.code_distance, self.num_qubits)]
        
        circuit.append('R', qreg)
        if self.initial_state == 1:
            circuit.append('X', data_qreg)
        circuit.append('H', qreg)
        
        # Sammanfläta qubits
        for qb in data_qreg[1:]:
            circuit.append('CNOT', [data_qreg[0], qb])
        
        # genomför syndrommätning för varje d_t
        for time_step in range(self.time_steps):
            circuit.append_operation("DEPOLARIZE1", data_qreg, self.error_rate)
            
            for qb in range(self.num_qubits-self.code_distance):
                circuit.append('CNOT', [qb+self.code_distance, qb])
                circuit.append('CNOT', [qb+self.code_distance, qb+1])
            
            circuit.append('H', ancilla_qreg)
            circuit.append("DEPOLARIZE1", ancilla_qreg, self.error_rate)
            circuit.append('MR', ancilla_qreg)
            circuit.append('H', ancilla_qreg)        
        
        # Slutlig mätninga av final_state
        circuit.append('H', data_qreg)
        circuit.append("DEPOLARIZE1", data_qreg, self.error_rate)
        circuit.append('M', data_qreg)
        
        return circuit

    def execute(self) -> object:
        """ Samplar kretsen och returnar syndrom och final state. """
        circuit = self.build_error_correction_sequence()
        
        # Print circuit in terminal. Works best for small d, dt.
        #print(circuit.diagram('timeline-text'))

        # Sample using Stim sampler
        sampler = circuit.compile_sampler()
        shots = sampler.sample(shots=self.shots).astype(int)

        # Format data to the same format as qiskit
        ancilla_bits = self.num_qubits - self.code_distance
        ancilla_meas = shots[:, :ancilla_bits*self.time_steps]
        data_meas = shots[:, ancilla_bits*self.time_steps:]
        syndrome = [''.join(str(bit) for bit in row) for row in ancilla_meas]
        final_state = [''.join(str(bit) for bit in row) for row in data_meas]
        
        return syndrome, final_state
            

if __name__ == "__main__":
    for d in [3, 5, 7, 9, 11]:
        for dt in [3, 5, 7, 9, 11]:
            result_dict = {}
            result_dict['results'] = [{}]
            result_dict['results'][0]['data'] = {}
            result_dict['results'][0]['data']['syndrome'] = []
            result_dict['results'][0]['data']['final_state'] = []
            for error_rate in [0.01, 0.05, 0.1, 0.15]:
                qec = QuantumErrorCorrectionStim(code_distance=d, time_steps=dt, shots=25000, initial_state=0, error_rate=error_rate)
                
                syndrome, final_state = qec.execute()
                
                # Save as json
                result_dict['results'][0]['data']['syndrome'].extend(syndrome)
                result_dict['results'][0]['data']['final_state'].extend(final_state)
                
            filename = f"./jobs/Stim_data/stim_{qec.code_distance}_{qec.time_steps}_{qec.shots*4}_{qec.initial_state}.json"
            with open(filename, "w") as file:
                json.dump(result_dict, file)
            
            print(f"Output sparad som '{filename}'.")
    