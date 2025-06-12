import numpy as np
import pymatching
import json
import time
import torch

class repetition_code_MWPM:
    '''
    MWPM decoder for the repetition code. Can use constant weights or custom weights
    from a weight matrix
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

    def MWPM(self, ratio, weight = 'p_ij', init_state='0'):
        '''
        MWPM decoder for the repetition code.

        
        PARAMETERS

        weight : str 
            '1' or 'p_ij'. Type of weights to be used in the decoder. For constant weights
            use '1'. For custom weights use 'p_ij'.
        init_state: str
            '1' or '0'. In which state the qubits were initialized. Assumes all qubits were
            initlialized in either 1 or 0
        '''
        def syndrome():
            '''
            Opens the detector data and creates a syndrome matrix.
            '''
            with open(self.run+'_data/Detector_data/detector_dict_'+self.backend_name+'_'+str(self.code_distance)
                    +'_'+str(self.shots)+'_'+str(self.time_steps)+'_'+self.version+'.json' , 'r') as infile:
                data = json.load(infile)
            s_m = np.array(list(data.values()))
            s_n = []
            for shot in range(self.shots):
                s_d = s_m[shot].tolist()
                s_n.append(sum(s_d, []))
            self.number_of_nodes = []
            for shot in s_n:
                self.number_of_nodes.append(sum(shot))
            return s_n

        def outcome():
            '''
            Opens the outcome (final measurement) of the qubits.
            '''
            with open(self.run+'_data/Outcome_data/outcome_dict_'+self.backend_name+'_'+str(self.code_distance)
                    +'_'+str(self.shots)+'_'+str(self.time_steps)+'_'+self.version+'.json', 'r') as infile:
                data = json.load(infile)
            return np.array(list(data.values()))
    
    #Initialize matching using a custom weight matrix or constant weights:
        matching = pymatching.Matching()
        if weight == 'p_ij':
            with open(self.run+'_data/Error_matrix/error_matrix_t_'+self.backend_name+'_'+str(self.code_distance)
                            +'_'+str(self.shots)+'_'+str(self.time_steps)+'_'+self.version+'.json' , 'r') as infile:
                error_matrix_t = np.array(json.load(infile))
                error_matrix_t[error_matrix_t == 0] = 0.0000001
                error_matrix_t[error_matrix_t < 0] = 0.0000001
                weight_matrix = -np.log(error_matrix_t)
        if weight == '1':
            weight_matrix = np.ones(((self.time_steps+1)*(self.code_distance-1),(self.time_steps+1)*(self.code_distance-1)))

    #Add space-like edges:
        for i in range(0,(self.time_steps+1)*(self.code_distance-1),self.code_distance-1):
            matching.add_boundary_edge(i, weight=weight_matrix[i][i+1], fault_ids={0} ,merge_strategy='replace')
            for j in range(self.code_distance-2):
                matching.add_edge(i+j,i+j+1, weight=weight_matrix[i+j][i+j+1], fault_ids={j+1}, merge_strategy='replace')
            matching.add_boundary_edge(i+j+1, weight=weight_matrix[i+j][i+j+1], fault_ids={self.code_distance-1}, merge_strategy='replace')

    #Add nearest neighbour time-like edges:
        for i in range(0,(self.time_steps+1)*(self.code_distance-1)-self.code_distance+1,self.code_distance-1):
            for j in range(self.code_distance-2):
                matching.add_edge(i+j,i+j+self.code_distance-1, weight=weight_matrix[i+j][i+j+self.code_distance-1], merge_strategy='replace')
                matching.add_edge(i+j+1,i+j+1+self.code_distance-1, weight=weight_matrix[i+j+1][i+j+1+self.code_distance-1], merge_strategy='replace')

    #Choose initial state and decode:
        graphs = syndrome()
        len_graphs = len(graphs)
        len_train = int(len_graphs*ratio)
        len_test = len_graphs - len_train

        graphs_train, graphs_test = torch.utils.data.random_split(
            graphs, [len_train, len_test], generator=torch.Generator().manual_seed(42)
        )
        train_indices = graphs_train.indices
        test_indices = graphs_test.indices
        
        # print(train_indices)
        # print(len(graphs_test))
        # print(len_test)
        
        time_start = time.perf_counter()
        if init_state == '1':
            prediction = (matching.decode_batch(graphs_test)+np.ones(self.code_distance, dtype=int))%2
        if init_state == '0':
            prediction = matching.decode_batch(graphs_test)
        self.duration_per_shot = (time.perf_counter() - time_start)/len_test


    #Calculate the logical error rate:
        success = 0
        type1 = 0
        type2 = 0
        true_p = 0
        true_n = 0
        outcome_m = outcome()
            
        for i in range(len_test):
            if self.number_of_nodes[test_indices[i]] == 0:
                success = success + 1
            elif self.number_of_nodes[test_indices[i]] != 0:     
                if prediction[i][-1] == outcome_m[test_indices[i]][-1]: # Fattar fortfarande inte varför man bara jämför med första biten...
                    success = success + 1
                



            logical_outcome = str(outcome_m[test_indices[i]][-1])
            logical_corrected =  str(prediction[i][-1])
               
            true_p += int(logical_outcome == '1' and logical_corrected == '1')
            true_n += int(logical_outcome == '0' and logical_corrected == '0')
            type1 += int(logical_outcome == '0' and logical_corrected == '1')
            type2 += int(logical_outcome == '1' and logical_corrected == '0')
                    
        self.true_p = true_p/(len_test)
        self.true_n = true_n/(len_test)
        self.type1 = type1/(len_test)
        self.type2 = type2/(len_test)        
        
        self.failrate = 1-(success/(len_test))