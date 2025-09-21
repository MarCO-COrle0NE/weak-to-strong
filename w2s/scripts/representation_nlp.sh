#!/bin/bash
#SBATCH --job-name=representation_nlp
#SBATCH --array=1
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=neu330
#SBATCH --time 4:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%A_%a.out            # error file name (same to watch just one file)

module load anaconda3/2024.02
export HF_HOME=/scratch/jl4476
export TORCH_HOME=/scratch/jl4476
source $HOME/.bashrc
#export PATH=/scratch/yl9315/miniconda3/bin:$PATH
conda activate Domain

# Deterministic
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$((10000 + ($SLURM_JOBID % 10000) + ($SLURM_ARRAY_TASK_ID % 1000)))

# Define your input file lists
TC=("gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl")
TC=("meta-llama/Meta-Llama-3.1-8B" "google/gemma-2-2b" "Qwen/Qwen2.5-1.5B" "microsoft/phi-2" "meta-llama/Llama-3.2-1B")

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(16)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  
INPUT_TC=${TC[$TASK_ID]}
SEED=0

mkdir -p /scratch/jl4476/data
rsync -av --ignore-existing ./domainbed/data/MNIST/ /scratch/$USER/data/

python3 -m domainbed.scripts.representation_nlp\
     --data_dir=/scratch/jl4476/data/SST2/\
     --dataset SST2\
     --model_name $INPUT_TC\
     --hparams "{\"lr\":1e-5, \"batch_size\":2, \"weight_decay\":0.0, \"warmup_steps\":0,\"gradient_accumulation_steps\":4}"\
     --n 100\
     --N 100\
     --test_length 4349\
     --epoch 2\
     --test_envs 2\
     --seed $SEED\
     --trial_seed $SEED\
     --output_dir /scratch/jl4476/${INPUT_DATASET}/representation\
