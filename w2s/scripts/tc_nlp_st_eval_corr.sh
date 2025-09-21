#!/bin/bash
#SBATCH --job-name=eval_corr_001
#SBATCH --array=3
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 48:00:00              # maximum execution time (HH:MM:SS)
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
#TC=("prajjwal1/bert-tiny" "prajjwal1/bert-mini" "prajjwal1/bert-small" "prajjwal1/bert-medium")
TC=("prajjwal1/bert-tiny" "prajjwal1/bert-mini" "prajjwal1/bert-small" "prajjwal1/bert-medium" "bert-base-uncased" "albert-base-v1" "albert-base-v2" "roberta-base")
TC=("prajjwal1/bert-tiny" "prajjwal1/bert-mini" "prajjwal1/bert-small" "prajjwal1/bert-medium" "bert-base-uncased" "albert-base-v1" "albert-base-v2" "roberta-base" "distilbert-base-uncased" "distilroberta-base" "google/electra-small-discriminator" "google/electra-base-discriminator")
#TC=("prajjwal1/bert-tiny")  # 3 teachers
# ST=("prajjwal1/bert-medium" "bert-base-uncased" "bert-large-uncased" "roberta-base" "roberta-large")
ST=("roberta-large")
#ST=("bert-base-uncased")
ST=("google/electra-base-discriminator")

DWS=(7000 8500 8500 4000 600 1000 600)
DWS=(7000 8500 8500 4000 600 4000 4000 600 4000 4000 4000 4000)
DWS=(7000 8500 8500 4000 500 8000 800 500 500 500 3500 700)
#DSS=(4000 600 1050 600 1500)
DSS=(600)
DSS=(700)

#DWS=(70 85 85 40 6 10)
#DSS=(6)

# DWS=(140 170 170 80)
# DSS=(80 12 21 12 30)
# DWS=(70 85 70 40 5 80 8 5 5 5 35 7)
# DSS=(7)

# DWS=(70 85 70 40 5 80 8 5 5 5 35 7)
# DSS=(7)

# DWS=(100 100 70 60 40 100 40 40 40 40 70 40)
# DSS=(40)

# DWS=(100 100 100 60 30 100 30 30 30 30 70 30)
# DSS=(30)
# DWS=(10 10 10 10 10 10 10 10 10 10 10 10)
# DSS=(10)
DATASET=("SST2") # 3 datasets

# Calculate the total number of combinations
# TOTAL_COMBINATIONS=$(24)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_TC=$((TASK_ID / ${#ST[@]})) # Get index for ST
IDX_ST=$((TASK_ID % ${#ST[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_DATASET="SST2"
INPUT_TC=${TC[$IDX_TC]}
INPUT_ST=${ST[$IDX_ST]}
DW=${DWS[$IDX_TC]}
DS=${DSS[$IDX_ST]}
SEED=0
INPUT_N=${SUBSET_ST[$IDX_SUBSET_ST]}

mkdir -p /scratch/jl4476/data/SST2/
rsync -av --ignore-existing ./domainbed/data/SST-2/ /scratch/$USER/data/SST2/

# python3 -m domainbed.cor_dim\
#        --data_dir=/scratch/jl4476/data/SST2/\
#        --model_name_s $INPUT_ST\
#        --model_name_w $INPUT_TC\
#        --d_s $DS\
#        --d_w $DW\
#        --transform_freq 4\

# python3 -m domainbed.cor_dim_batch_32\
#        --data_dir=/scratch/jl4476/data/SST2/\
#        --model_name_s $INPUT_ST\
#        --model_name_w $INPUT_TC\
#        --d_s $DS\
#        --d_w $DW\
#        --common_D 2000000\
#        --transform_freq 4\
#        --runs 6\

# python3 -m domainbed.cor_dim_batch_32\
#        --data_dir=/scratch/jl4476/data/SST2/\
#        --model_name_s $INPUT_ST\
#        --model_name_w $INPUT_TC\
#        --d_s $DS\
#        --d_w $DW\
#        --transform_freq 8\
#        --runs 10\

python3 -m domainbed.cor_dim_batch_32\
       --data_dir=/scratch/jl4476/data/SST2/\
       --model_name_s $INPUT_ST\
       --model_name_w $INPUT_TC\
       --d_s $DS\
       --d_w $DW\
       --transform_freq 8\
       --select_percentage 0.01\
       --runs 10\
       