import json
from qiskit_ibm_runtime import RuntimeDecoder
from qiskit.primitives.containers import PrimitiveResult
import numpy as np

def job_data_formatter(run,backend_name,code_distance,shots,time_steps,version):

    with open(run + '_data/Job_data/'+backend_name+'_'+str(code_distance)+'_'
             +str(time_steps)+'_'+str(shots)+'_'+version+'.json', "r") as file:
        result = json.load(file, cls=RuntimeDecoder)
    
    if isinstance(result, PrimitiveResult): 
        result = result[0] #Kan behövas ibland om PrimitiveResult error

    syndrome = result.data.syndrome.get_bitstrings()
    final_state = result.data.final_state.get_bitstrings()
    
    # Reverse strings
    final_state = [x[::-1] for x in final_state]
    syndrome = [x[::-1] for x in syndrome]
    
    raw_data = [final_state]
    for j in range(0, time_steps*(code_distance-1), code_distance-1):
        temp_array = []
        for i in range(shots):
            temp_array.append(syndrome[i][j:(j+code_distance-1)])
        raw_data.append(temp_array)
        
    json_object = json.dumps(raw_data)
    with open(run + '_data/Raw_data/result_matrix_'+backend_name+'_'+str(code_distance)+'_'
             +str(shots)+'_'+str(time_steps)+'_'+version+'.json', 'w') as outfile:
        outfile.write(json_object)
        
        
# run,backend_name,code_distance,shots,time_steps,version = "C:/Users/libuf/Downloads/MWPM/Från Mats/hej", "czhftbszj67g008skcj0", 3, 2000000, 5, "0" 
# job_data_formatter(run,backend_name,code_distance,shots,time_steps,version)