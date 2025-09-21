import os
import subprocess
import json

# List of seeds
SEEDS = [6,8] #[42, 123, 456, 789]  # Replace with your desired seeds
NAMES = ['var','var']

job_ids_dict = {
    "tc": [
        "1880639",
        "1880640"
    ],
    "st": [
        "1880641",
        "1880642"
    ],
    "tc_ceil": [
        "1880643",
        "1880644"
    ],
    "ceil": [
        "1880607"
    ]
}

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

def collect_tc(job_number,output_file='tc.json',job_name='tc',seed=0):
    command = ["python","collect_tc_nlp_var.py",f"--job_number={job_number}",f"--job_name={job_name}",f"--seed={seed}",f"--output_file={output_file}"]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
        #job_id = result.stdout.strip().split()[-1]  # Extract job ID from sbatch output
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        return None

def collect_tc_ceil(job_number,output_file='tc.json',job_name='tc',seed=0):
    command = ["python","collect_tc_ceil_nlp_var.py",f"--job_number={job_number}",f"--job_name={job_name}",f"--seed={seed}",f"--output_file={output_file}"]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
        #job_id = result.stdout.strip().split()[-1]  # Extract job ID from sbatch output
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        return None

def collect_st(job_number,output_file='st.json',job_name='st',st="google/electra-base-discriminator",seed=0):
    command = ["python","collect_st_nlp_var.py",f"--job_number={job_number}",f"--job_name={job_name}",f"--seed={seed}",f"--output_file={output_file}",f"--st={st}"]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
        #job_id = result.stdout.strip().split()[-1]  # Extract job ID from sbatch output
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        return None

def collect_ceil(job_number,output_file='ceil.json',job_name='ceil',st="google/electra-base-discriminator",seed=0):
    command = ["python","collect_ceil_nlp_var.py",f"--job_number={job_number}",f"--job_name={job_name}",f"--seed={seed}",f"--output_file={output_file}",f"--st={st}"]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
        #job_id = result.stdout.strip().split()[-1]  # Extract job ID from sbatch output
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        return None

for (tc_id,seed,name) in zip(job_ids_dict['tc'],SEEDS,NAMES):
    result = collect_tc(tc_id,output_file=f'scripts/nlp/tc_var_{name}.json',job_name=f'tc_{name}',seed=seed)
    print(result)

for (tc_id,seed,name) in zip(job_ids_dict['tc_ceil'],SEEDS,NAMES):
    result = collect_tc_ceil(tc_id,output_file=f'scripts/nlp/tc_var_{name}.json',job_name=f'tc_ceil_{name}',seed=seed)
    print(result)

for (st_id,seed,name) in zip(job_ids_dict['st'],SEEDS,NAMES):
    result = collect_st(st_id,output_file=f'scripts/nlp/st_var_{name}.json',job_name=f'st_{name}',seed=seed)
    print(result)

if 'ceil' in job_ids_dict:
    for (ceil_id,seed,name) in zip(job_ids_dict['ceil'],SEEDS[:1],NAMES[:1]):
        result = collect_ceil(ceil_id,output_file=f'scripts/nlp/ceil_var_{name}.json',job_name=f'ceil_{name}',seed=seed)
        print(result)

print("All jobs submitted successfully.")
# with open(f'scripts/nlp/job_ids_2.json','w') as file:
#     json.dump(job_ids_dict,file,indent=4)