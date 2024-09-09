# main_script.py
import subprocess
import numpy as np
from multiprocessing import Pool, cpu_count

expName = "experiment_runPatientsDTI_good_brats.py"

def process_patient(patientID):
    result = subprocess.run(['python', expName, str(patientID)], capture_output=True, text=True)
    print(f"Output for patient {patientID}: {result.stdout}")
    if result.stderr:
        print(f"Error for patient {patientID}: {result.stderr}")

if __name__ == '__main__':
    patients =np.arange(116, 500, 1) # 180  [115]#
    print(patients)
    
    # Determine the number of worker processes to use
    num_workers = 10 #10

    # Create a pool of worker processes
    with Pool(processes=num_workers) as pool:
        results = [pool.apply_async(process_patient, args=(patientID,)) for patientID in patients]
        for result in results:
            result.get() 
