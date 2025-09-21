#!/bin/bash
#SBATCH --job-name=tc_nlp_said3
#SBATCH --array=1-4
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 24:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%A_%a.out            # error file name (same to watch just one file)

module load anaconda3/2024.02
export HF_HOME=/scratch/jl4476
export TORCH_HOME=/scratch/jl4476
source $HOME/.bashrc
#export PATH=/scratch/yl9315/miniconda3/bin:$PATH
conda activate Domain

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Deterministic
#export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Define your input file lists
TC=("bert-base-uncased" "bert-large-uncased" "roberta-base" "roberta-large")  # 3 teachers
#TC=("prajjwal1/bert-tiny" "prajjwal1/bert-mini" "prajjwal1/bert-small" "prajjwal1/bert-medium" "bert-base-uncased" "bert-large-uncased" "microsoft/deberta-v3-small")
#SUBSET=(0.00273 0.0036 0.005 0.01 0.015 0.02 0.02322)
#SUBSET_INT=(75 100 140 280 420 560 650)
#SUBSET=(0.00714 0.02143 0.03571 0.05 0.0643)
#SUBSET_INT=(200 600 1000 1400 1800)
SUBSET=(0.001)
SUBSET_INT=(28)
#SUBSET=(0.5 0.25 0.125)  # 3 students
#DATASET=("SST2") # 3 datasets
# INTR_DIMS=(800 900 1000 1200 1500)
TC=("bert-base-uncased" "roberta-base")
INTR_DIMS=(500 550)

# TC=("bert-large-uncased" "roberta-large")
# INTR_DIMS=(1050 1200 1300)

# Calculate the total number of combinations
# TOTAL_COMBINATIONS=$(24)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_TC=$((TASK_ID / ${#INTR_DIMS[@]})) # Get index for ST
IDX_DIM=$((TASK_ID % ${#INTR_DIMS[@]}))    # Get index for DATASET
IDX_SUBSET=$IDX_TC

# Get the corresponding input files
INPUT_DATASET="SST2"
INPUT_SUBSET=${SUBSET[$IDX_SUBSET]}
INPUT_SUBSET_INT=58000
INPUT_TC=${TC[$IDX_TC]}
SEED=0
INTR_DIM=${INTR_DIMS[$IDX_DIM]}

mkdir -p /scratch/jl4476/data/SST2/
rsync -av --update ./domainbed/data/SST-2/ /scratch/$USER/data/SST2/

# python3 -m domainbed.scripts.tc_nlp\
#        --data_dir=/scratch/jl4476/data/SST2/\
#        --dataset ${INPUT_DATASET}\
#        --model_name $INPUT_TC\
#        --hparams "{\"lr\":1e-3, \"batch_size\":32, \"momentum\":0.9}"\
#        --n $INPUT_SUBSET_INT\
#        --N 1000\
#        --test_length 10000\
#        --epoch 12\
#        --test_envs 2\
#        --seed $SEED\
#        --trial_seed $SEED\
#        --intrinsic_dimension $INTR_DIM\
#        --said\
#        --SGD\
#        --output_dir /scratch/jl4476/${INPUT_DATASET}/tc_said/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INTR_DIM}\

python3 -m domainbed.scripts.tc_nlp\
       --data_dir=/scratch/jl4476/data/SST2/\
       --dataset ${INPUT_DATASET}\
       --model_name $INPUT_TC\
       --hparams "{\"lr\":1e-4, \"batch_size\":32, \"warmup_steps\":100}"\
       --n $INPUT_SUBSET_INT\
       --N 1000\
       --test_length 7349\
       --epoch 12\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --intrinsic_dimension $INTR_DIM\
       --said\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/tc_said_adamw/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INTR_DIM}\

mkdir -p /auto/u/jl4476/w2s/${INPUT_DATASET}/tc_said/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INTR_DIM}/
cp -rf /scratch/jl4476/${INPUT_DATASET}/tc_said/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INTR_DIM}/logs /auto/u/jl4476/w2s/${INPUT_DATASET}/tc_said/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INTR_DIM}/