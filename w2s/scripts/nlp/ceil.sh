#!/bin/bash
#SBATCH --job-name=ceil_0
#SBATCH --array=1-63
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 8:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%A_%a.out            # error file name (same to watch just one file)
#SBATCH --export=SEED

module load anaconda3/2024.02
export HF_HOME=/scratch/jl4476
export TORCH_HOME=/scratch/jl4476
source $HOME/.bashrc
#export PATH=/scratch/yl9315/miniconda3/bin:$PATH
conda activate Domain

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
# MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
#MASTER_PORT=$(expr 10000 + $(echo -n $((SLURM_JOBID + $(SLURM_ARRAY_TASK_ID))) | tail -c 4))
#MASTER_PORT=$((10000 + $(printf "%04d" $((SLURM_JOBID + SLURM_ARRAY_TASK_ID)) | tail -c 4)))
# Generate a unique port per SLURM task using job ID and task ID
MASTER_PORT=$((10000 + ($SLURM_JOBID % 10000) + ($SLURM_ARRAY_TASK_ID % 1000)))

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

export NCCL_DEBUG=INFO               # Enable NCCL debugging (optional)
export PYTHONFAULTHANDLER=1          # Enable Python traceback on fault

export TOKENIZERS_PARALLELISM=false
# Deterministic
#export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Define your input file lists
#TC=("prajjwal1/bert-tiny" "prajjwal1/bert-mini" "prajjwal1/bert-small" "prajjwal1/bert-medium")
TC=("prajjwal1/bert-tiny" "prajjwal1/bert-mini" "prajjwal1/bert-medium")  # 3 teachers
#TC=("bert-base-uncased" "albert-base-v1")
TC=("google/electra-base-discriminator")
SUBSET=(4000 8000 12000 16000)
SUBSET_ST=(20000 28000 36000 44000)

SUBSET=(500 1000 2000 3000)
SUBSET_ST=(3000 5000 10000 50000)

SUBSET=(500 1000 2000 3000 4000 8000 12000)
SUBSET=(100 300 500 1000 2000 4000 8000)
SUBSET_ST=(16000 24000 32000 40000 48000)

SUBSET=(100 300 500 1000 2000 4000 8000 12000 14000 16000 20000)
SUBSET_ST=(12000 14000 16000 18000 24000 32000 40000)

SUBSET=(300 1000 6000 13000 20000 23000)
SUBSET_ST=(8000 14000 20000 25000 30000 40000)

SUBSET=(4000 6000 7000 8000 9000 10000 12000 13000)
SUBSET_ST=(5000 9000 10000 11000 12000 13000 14000 15000 17000 20000 25000 30000)

SUBSET=(8500 9500 11000 14000)
SUBSET_ST=(30000)

SUBSET=(8000 9000 10000 11000 12000 13000 14000 15000)
SUBSET_ST=(30000)
# SUBSET=(10000)
# SUBSET_ST=(15000 17500 20000 22500 25000 27500 30000)
SUBSET=(4000 6000 7000 8000 9000 10000 12000 13000 15000)
SUBSET_ST=(8000 15000 20000 25000 30000 35000 40000)

DATASET=("SST2") # 3 datasets

# Calculate the total number of combinations
# TOTAL_COMBINATIONS=$(24)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_SUBSET=$((TASK_ID / ${#SUBSET_ST[@]})) # Get index for ST
IDX_SUBSET_ST=$((TASK_ID % ${#SUBSET_ST[@]}))    # Get index for DATASET
# IDX_TC=$((TASK_ID / $((${#SUBSET[@]} * ${#SUBSET_ST[@]})))) # Get index for ST
# IDX_SUBSET=$(($((TASK_ID % $((${#SUBSET[@]} * ${#SUBSET_ST[@]})))) / ${#SUBSET_ST[@]})) # Get index for TC
# IDX_SUBSET_ST=$((TASK_ID % ${#SUBSET_ST[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_DATASET="SST2"
INPUT_SUBSET=${SUBSET[$IDX_SUBSET]}
INPUT_SUBSET_INT=${SUBSET[$IDX_SUBSET]}
INPUT_N=${SUBSET_ST[$IDX_SUBSET_ST]}
INPUT_SUBSET_ST=$INPUT_N
#INPUT_TC=${TC[$IDX_TC]}
#INPUT_ST="bert-base-uncased"
INPUT_ST="google/electra-base-discriminator"
#SEED=0
INPUT_WARMUP=$((($INPUT_N + $INPUT_SUBSET_INT) / 64))

mkdir -p /scratch/jl4476/data/SST2/
rsync -av --update ./domainbed/data/SST-2/ /scratch/$USER/data/SST2/

python3 -m domainbed.scripts.ceil_nlp\
       --data_dir=/scratch/jl4476/data/SST2/\
       --dataset ${INPUT_DATASET}\
       --model_name $INPUT_ST\
       --hparams "{\"lr\":5e-5, \"batch_size\":32, \"weight_decay\":0.0, \"warmup_steps\":0}"\
       --n $INPUT_SUBSET_INT\
       --N $INPUT_N\
       --test_length 4349\
       --epoch 8\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/ceil/${SEED}/${INPUT_SUBSET_INT}_${INPUT_N}_${INPUT_ST}\

#rsync -av --ignore-existing /scratch/$USER/data/SST2/ ./domainbed/data/SST-2/ 

# accelerate launch --multi_gpu --num_processes 2 --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT -m domainbed.scripts.ceil_nlp_acc\
#        --data_dir=/scratch/jl4476/data/SST2/\
#        --dataset ${INPUT_DATASET}\
#        --model_name $INPUT_ST\
#        --hparams "{\"lr\":5e-5, \"batch_size\":16, \"weight_decay\":0.0, \"warmup_steps\":${INPUT_WARMUP}}"\
#        --n $INPUT_SUBSET_INT\
#        --N $INPUT_N\
#        --test_length 4349\
#        --epoch 5\
#        --test_envs 2\
#        --seed $SEED\
#        --trial_seed $SEED\
#        --output_dir /scratch/jl4476/${INPUT_DATASET}/ceil/${SEED}/${INPUT_SUBSET_INT}_${INPUT_N}_${INPUT_ST}\

mkdir -p /auto/u/jl4476/w2s/${INPUT_DATASET}/ceil/${SEED}/${INPUT_SUBSET_INT}_${INPUT_SUBSET_ST}_${INPUT_ST}/
cp -rf /scratch/jl4476/${INPUT_DATASET}/ceil/${SEED}/${INPUT_SUBSET_INT}_${INPUT_SUBSET_ST}_${INPUT_ST}/logs /auto/u/jl4476/w2s/${INPUT_DATASET}/ceil/${SEED}/${INPUT_SUBSET_INT}_${INPUT_SUBSET_ST}_${INPUT_ST}/