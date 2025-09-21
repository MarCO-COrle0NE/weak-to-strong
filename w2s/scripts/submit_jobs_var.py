import os
import subprocess
import json

# List of seeds
SEEDS = [6,8] #[42, 123, 456, 789]  # Replace with your desired seeds

# Paths to your .sh files
SCRIPT_1 = "scripts/nlp/tc_var.sh"  # Path to the first script
SCRIPT_2 = "scripts/nlp/st_var.sh"  # Path to the second script
#SCRIPT_5 = "scripts/nlp/st1.sh"
SCRIPT_3 = "scripts/nlp/tc_ceil_var.sh"
SCRIPT_4 = "scripts/nlp/ceil_var.sh"

job_ids_dict = {}

# Function to submit a job and return its job ID
def submit_job(script, seed=None, depend_job_id=None):
    command = ["sbatch"]
    if seed is not None:
        command.extend(["--export=SEED={}".format(seed)])
    if depend_job_id is not None:
        command.extend(["--dependency=afterok:{}".format(depend_job_id).format(seed)])
    command.append(script)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]  # Extract job ID from sbatch output
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        return None

# Submit all array jobs for 1.sh
job_ids_1 = []
for seed in SEEDS:
    job_id = submit_job(SCRIPT_1, seed)
    if job_id:
        job_ids_1.append(job_id)
        print(f"Submitted {SCRIPT_1} with seed={seed}, job ID={job_id}")
job_ids_dict['tc'] = job_ids_1

# Submit 2.sh with dependencies on 1.sh
job_ids_2 = []
for seed, job_id_1 in zip(SEEDS, job_ids_1):
    job_id_2 = submit_job(SCRIPT_2, seed, depend_job_id=job_id_1)
    if job_id_2:
        job_ids_2.append(job_id_2)
        print(f"Submitted {SCRIPT_2} with seed={seed}, job ID={job_id_2}")
        # try:
        #     # Set dependency for 2.sh to afterok of the corresponding 1.sh job
        #     subprocess.run(["scontrol", "update", "jobid={}".format(job_id_2), "dependency=afterok:{}".format(job_id_1)], check=True)
        #     print(f"Submitted {SCRIPT_2} with seed={seed}, job ID={job_id_2}, depends on job ID={job_id_1}")
        # except subprocess.CalledProcessError as e:
        #     print(f"Error setting dependency for job {job_id_2}: {e}")
job_ids_dict['st'] = job_ids_2

# job_ids_5 = []
# for seed, job_id_1 in zip(SEEDS, job_ids_1):
#     job_id_5 = submit_job(SCRIPT_5, seed, depend_job_id=job_id_1)
#     if job_id_5:
#         job_ids_5.append(job_id_5)
#         print(f"Submitted {SCRIPT_5} with seed={seed}, job ID={job_id_5}")
# job_ids_dict['st1'] = job_ids_5

job_ids_3 = []
for seed in SEEDS:
    job_id = submit_job(SCRIPT_3, seed)
    if job_id:
        job_ids_3.append(job_id)
        print(f"Submitted {SCRIPT_3} with seed={seed}, job ID={job_id}")
job_ids_dict['tc_ceil'] = job_ids_3


# job_ids_4 = []
# for seed in [6]:
#     job_id = submit_job(SCRIPT_4, seed)
#     if job_id:
#         job_ids_4.append(job_id)
#         print(f"Submitted {SCRIPT_4} with seed={seed}, job ID={job_id}")
# job_ids_dict['ceil'] = job_ids_4

print("All jobs submitted successfully.")
with open(f'scripts/nlp/job_ids_var2.json','w') as file:
    json.dump(job_ids_dict,file,indent=4)