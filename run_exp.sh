#!/bin/bash

# List of patient IDs to process
patient_ids=("14")
# , "70", "101", "131", "115")

# Path to your Python script
script_path="experiment_runPatientsDTI_good_brats.py"

# Loop through each patient ID and run the Python script in background
for patient_id in "${patient_ids[@]}"
do
  echo "Running script for patient ID: $patient_id"
  python3 $script_path $patient_id &
done

# Wait for all background processes to complete
wait