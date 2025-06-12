import json
import numpy as np


class repetition_code_data:
    '''
    Formatter and error rate calculator for repetition code data from qiskit runs.
    Assumes the following directory structure and file name convention:

    'run'_data/
        Detector_data/ 
            detector_dict_...
        Error_matrix/
            error_matrix_q_...
            error_matrix_t_...
        Format_data/
            result_dict_...
        Outcome_data/
            outcome_dict_...
        Raw_data/
            result_matrix_...

    Note that 'run' can be freely choosen by the user, just note that the same 'run'
    is used in the code the find the files. The ellipses are replaced by the following
    string:
        'backend_name'_'code_distance'_'shots'_'time_steps'_'version'.json
    Note that this file structure and naming convention is simply used to keep track
    of the different measurements and can be easily changed by the user (simply change
    the names where files are opened and correspondingly the __init__ parameters).

    It is recommended to run the code and create the directories in a virtual environment.

    (Note that this directory structure has been replaced by a better one in the
    surface code code and this one should be changed to follow the same one.)
    '''

    def __init__(self,run,backend_name,code_distance,shots,time_steps,version):
        '''
        PARAMETERS

        run : str 
            Name of run (only used in the loading and naming of files)
        backend_name: str
            Name of backend (only used in the loading and naming of files)
        code_distance: int
            Code distance of the repetition code
        shots: int
            Number of shots when collecting the data
        time_steps: int
            Number of time repetitions
        version: str
            Version (only used in the loading and naming of files)
        '''

        self.run = run
        self.backend_name = backend_name
        self.code_distance = code_distance
        self.shots = shots
        self.time_steps = time_steps
        self.version = version

    def format(self):
        '''
        Format the raw qiskit data to a format that can be used by repetition_code_MWPM (and STIM).

        The raw data from qiskit is assumed to be a matrix of size time_steps * shots and with
        the following form:
            [ [final qubit state measurement]      ,
              [ancilla measurement at time_step 1] , 
              [ancilla measurement at time_step 2] , 
                            ...                      ].
            
        All qubit states are assumed to be saved as strings, i.e 'q1 q2 q3 ...'. The same is
        true for the ancilla states, i.e 'a1 a2 a3 ...'.

        As an example, a measurement for code distance 3 with 5 shots and 3 time repetitions could
        look like:
            [ ['001', '000', '000', '010', '000'],
              ['00', '00', '00', '00', '00']     , 
              ['01', '00', '10', '10', '00']     , 
              ['01', '00', '00', '11', '00']     , 
              ['01', '00', '00', '01', '00']     , 
              ['01', '00', '00', '11', '00']       ].
        
        The first row contains the final qubit measurements and the strings are of length 3. The other rows
        contain ancilla measurements and the strings are of length 3-1=2.
        '''
        with open(self.run + '_data/Raw_data/result_matrix_'+self.backend_name+'_'+str(self.code_distance)+'_'
                +str(self.shots)+'_'+str(self.time_steps)+'_'+self.version+'.json', 'r') as infile:
            raw_data = json.load(infile)
        ancilla = self.code_distance-1

        def final_stab_state():
            '''
            Calculate the final stabilizers from the final qubit measurement.
            '''
            final_state = raw_data[0]
            stab_matrix = []
            for i in range(self.shots):
                stab_string = ''
                for j in range (0, self.code_distance-1):
                    stab_value = np.bitwise_xor(int(final_state[i][j]), int(final_state[i][j+1]))
                    stab_string += str(stab_value)
                stab_matrix.append(stab_string)
            return stab_matrix

        def first_stab_state():
            '''
            Calculate the initial stabilizers from the inital qubit states.
            '''
            return(np.full(self.shots, '0'*ancilla))

        def stab_matrix():
            '''
            Create a matrix of all stabilizer measurements. Stabilizer space-like position is the column index
            and the time-like position is the row index. Every shot is stored in a separate matrix that are
            in turn stored in a dictionary.
            '''
            all_measurement_matrix = []
            undim_matrix = raw_data[1:]
            undim_matrix.insert(0, list(first_stab_state()))
            undim_matrix.append(list(final_stab_state()))
            for t in range(self.time_steps+2):
                all_measurement_matrix.append(list(map(int,list(''.join(undim_matrix[t])))))
            all_measurement_matrix = np.array(all_measurement_matrix)
            left_shift_amm = all_measurement_matrix[1:]
            right_shift_amm = all_measurement_matrix[:len(all_measurement_matrix)-1]
            all_detector_matrix = np.bitwise_xor(left_shift_amm, right_shift_amm)
            measurement_dict = {}
            outcome_dict = {}
            detector_dict = {}
            for m in range(1,self.shots+1):
                measurement_shot = all_measurement_matrix[:, (m-1)*ancilla:m*ancilla]
                measurement_dict[m-1] = measurement_shot.tolist()
                detector_shot = all_detector_matrix[:, (m-1)*ancilla:m*ancilla]
                detector_dict[m-1] = detector_shot.tolist()
                outcome_dict[m-1] = list(map(int, raw_data[0][m-1]))
            return measurement_dict, detector_dict, outcome_dict
        data = stab_matrix()

        #Save the data in json files:
        json_object = json.dumps(data[0])
        with open(self.run + '_data/Format_data/result_dict_'+self.backend_name+'_'+str(self.code_distance)
                +'_'+str(self.shots)+'_'+str(self.time_steps)+'_'+self.version+'.json', 'w') as outfile:
            outfile.write(json_object)
        json_object = json.dumps(data[1])
        with open(self.run + '_data/Detector_data/detector_dict_'+self.backend_name+'_'
                +str(self.code_distance)+'_'+str(self.shots)+'_'+str(self.time_steps)+'_'+self.version+'.json', 'w') as outfile:
            outfile.write(json_object)
        json_object = json.dumps(data[2])
        with open(self.run + '_data/Outcome_data/outcome_dict_'+self.backend_name+'_'
                +str(self.code_distance)+'_'+str(self.shots)+'_'+str(self.time_steps)+'_'+self.version+'.json', 'w') as outfile:
            outfile.write(json_object)
        return data



    def error_analyzer_q(self):
        '''
        Analyzes the frequency of errors and calculates a error correlation matrix. This uses
        qubit-first ordering. See error_analyzer_t for time-first ordering.
        
        (This code is very slow and has been vastly improved in the surface code formatter. The
        surface code code should be inserted here).
        '''
        with open(self.run + '_data/Detector_data/detector_dict_'+self.backend_name+'_'+str(self.code_distance)
                +'_'+str(self.shots)+'_'+str(self.time_steps)+'_'+self.version+'.json', 'r') as infile:
            data = json.load(infile)
            data = np.array(list(data.values()))

        def converter(data):
            '''
            Format data.

            PARAMETERS

            data: numpy.array
                Array of the detector data.
            '''
            new_data = []
            for shot in range(len(data)):
                temp_data = []
                for i in range(self.code_distance-1):
                    temp_data = temp_data + data[shot][:, i].tolist()
                new_data.append(temp_data)
            return(new_data)
        
        def X(data):
            '''
            Create self-correlation array.

            PARAMETERS

            data: numpy.array
                Array of the detector data.
            '''
            return np.sum(data, axis = 0)/len(data)

        def XX(data):
            '''
            Create correlation array.

            PARAMETERS

            data: numpy.array
                Array of the detector data.
            '''
            new_data = []
            for shot in range(len(data)):
                mat = np.zeros((len(data[0]),len(data[0])), dtype = int)
                for i in range(len(data[0])):
                    ref_value = data[shot][i]
                    for j in range(len(data[0])):
                        comp_value = data[shot][j]
                        mat[i][j] = ref_value*comp_value
                new_data.append(mat.tolist())
            return(np.sum(new_data,axis=0)/len(data))

        def P(XX, X):
            '''
            Calulate error correlations.

            PARAMETERS

            XX: list
                Correlation array
            X: list
                Self-correlation array
            '''
            XX = np.array(XX)
            X = np.array(X)
            mat = np.zeros_like(XX, dtype = float)
            for i in range(len(X)):
                for j in range(len(X)):
                    if (1-2*X[i])*(1-2*X[j]) == 0:
                        mat[i][j] = 10000
                    else:
                        mat[i][j] = (XX[i][j] - X[i]*X[j])/((1-2*X[i])*(1-2*X[j]))
                    if i == j:
                        mat[i][j] = 0
            return mat

        matrix = P(XX(converter(data)),X(converter(data)))
        json_object = json.dumps(matrix.tolist())
        with open(self.run + '_data/Error_matrix/error_matrix_q_'+self.backend_name+'_'+str(self.code_distance)
                +'_'+str(self.shots)+'_'+str(self.time_steps)+'_'+self.version+'.json', 'w') as outfile:
            outfile.write(json_object)
        return matrix



    def error_analyzer_t(self):
        '''
        Analyzes the frequency of errors and calculates a error correlation matrix. This uses
        time-first ordering. See error_analyzer_q for qubit-first ordering.
         
        (This code is very slow and has been vastly improved in the surface code formatter. The
        surface code code should be inserted here).
        '''
        with open(self.run + '_data/Detector_data/detector_dict_'+self.backend_name+'_'+str(self.code_distance)
                +'_'+str(self.shots)+'_'+str(self.time_steps)+'_'+self.version+'.json', 'r') as infile:
            data = json.load(infile)
            data = np.array(list(data.values()))

        def converter(data):
            '''
            Format data.

            PARAMETERS

            data: numpy.array
                Array of the detector data.
            '''
            new_data = []
            for shot in range(len(data)):
                temp_data = []
                for i in range(len(data[0])):
                    temp_data = temp_data + data[shot][i].tolist()
                new_data.append(temp_data)
            return(new_data)
        
        def X(data):
            '''
            Create self-correlation array.

            PARAMETERS

            data: numpy.array
                Array of the detector data.
            '''
            return np.sum(data, axis = 0)/len(data)

        def XX(data):
            '''
            Create correlation array.

            PARAMETERS

            data: numpy.array
                Array of the detector data.
            '''
            new_data = []
            for shot in range(len(data)):
                mat = np.zeros((len(data[0]),len(data[0])), dtype = int)
                for i in range(len(data[0])):
                    ref_value = data[shot][i]
                    for j in range(len(data[0])):
                        comp_value = data[shot][j]
                        mat[i][j] = ref_value*comp_value
                new_data.append(mat.tolist())
            return(np.sum(new_data,axis=0)/len(data))

        def P(XX, X):
            '''
            Calulate error correlations.

            PARAMETERS

            XX: list
                Correlation array
            X: list
                Self-correlation array
            '''
            XX = np.array(XX)
            X = np.array(X)
            mat = np.zeros_like(XX, dtype = float)
            for i in range(len(X)):
                for j in range(len(X)):
                    if (1-2*X[i])*(1-2*X[j]) == 0:
                        mat[i][j] = 10000
                    else:
                        mat[i][j] = (XX[i][j] - X[i]*X[j])/((1-2*X[i])*(1-2*X[j]))
                    if i == j:
                        mat[i][j] = 0
            return mat
        
#        print(np.array(converter(data))[:, 0])

        matrix = P(XX(converter(data)),X(converter(data)))
        json_object = json.dumps(matrix.tolist())
        with open(self.run + '_data/Error_matrix/error_matrix_t_'+self.backend_name+'_'+str(self.code_distance)
                +'_'+str(self.shots)+'_'+str(self.time_steps)+'_'+self.version+'.json', 'w') as outfile:
            outfile.write(json_object)
        return matrix