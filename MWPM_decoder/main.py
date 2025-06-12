from repetition_code_MWPM import repetition_code_MWPM
from repetition_code_data import repetition_code_data
from job_data_formatter import job_data_formatter

def failrate(run,backend_name,code_distance,shots,time_steps,version, MWPM_weights='1', initial_state='0', split_ratio = 0):
    '''
    PARAMETERS

    run : str 
        Name of run directory
    backend_name: str
        Name of backend directory
    code_distance: int
        Code distance in the job file
    shots: int
        Number of shots in the job file
    time_steps: int
        Number of time repetitions in the job file
    version: str
        Version (last character in jobfile name string)
    '''
    
    # Convert job data to correct matrix format for repitition_coda_data.py
    job_data_formatter(run,backend_name,code_distance,shots,time_steps,version)

    # Convert syndrom matrices to detector event matrices
    Data = repetition_code_data(run,backend_name,code_distance,shots,time_steps,version)
    Data.format()

    # Run MWPM algorithm for all shots and print failrate
    MWPM = repetition_code_MWPM(run,backend_name,code_distance,shots,time_steps,version)
    MWPM.MWPM(ratio = split_ratio, weight = MWPM_weights, init_state = initial_state)
    number_of_nodes = MWPM.number_of_nodes
    logical_failrate = MWPM.failrate
    true_positives = MWPM.true_p
    true_negatives = MWPM.true_n
    type1_errors = MWPM.type1
    type2_errors = MWPM.type2
    duration_per_shot = MWPM.duration_per_shot
    
    print(f"Average decoding time per shot: {duration_per_shot} seconds")
    print(f"Logical failrate: {logical_failrate*100.0}%")
    print(f"True positives: {true_positives*100.0}%")
    print(f"True negatives: {true_negatives*100.0}%")
    print(f"Type 1 errors: {type1_errors*100.0}%")
    print(f"Type 2 errors: {type2_errors*100.0}%")
    print(f"Logical failrate calculated using Type 1 + Type 2: {(type1_errors + type2_errors)*100.0}%")
    

# Examples

failrate('C:/Users/libuf/Documents/Kandidatarbete/MWPM/IBM', 'cyvk8tvy2gd000889470', 3, 3200, 5, '0', split_ratio=0.3)
# failrate('C:/Users/libuf/Documents/Kandidatarbete/MWPM/IBM', 'czjmrah7m0r0008wehf0', 5, 2000000, 5, '0', split_ratio=0.3)