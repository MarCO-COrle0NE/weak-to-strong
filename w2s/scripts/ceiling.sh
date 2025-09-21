#!/bin/bash
#SBATCH --job-name=ceiling_dinov2
#SBATCH --array=45-48
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 48:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%A_%a.out            # error file name (same to watch just one file)

module purge
source $HOME/.bashrc
export PATH=/scratch/yl9315/miniconda3/bin:$PATH
#module load cuda/11.6.2
conda activate DomainBed2

# Deterministic
export CUBLAS_WORKSPACE_CONFIG=:4096:8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Define your input file lists
#TC=("18")  # 3 teachers
#SUBSET=(1 0.02 0.05 0.07 0.1 0.2 0.5 0.75 1)
SEED=(1 2 3 4)
STS="dinov2"
ST=("s14" "b14" "l14" "g14")
SUBSET=(1)
DATASET=("CIFAR100" "CIFAR10" "Places365") 

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(8)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_DATASET=$((TASK_ID / $((${#ST[@]} * ${#SEED[@]})))) # Get index for ST
IDX_ST=$(($((TASK_ID % $((${#ST[@]} * ${#SEED[@]})))) / ${#SEED[@]})) # Get index for TC
IDX_SEED=$((TASK_ID % ${#SEED[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_DATASET=${DATASET[$IDX_DATASET]}
INPUT_SUBSET=${SUBSET[0]}
INPUT_ST=${ST[$IDX_ST]}
INPUT_SEED=${SEED[$IDX_SEED]}

python3 -m domainbed.scripts.ceiling\
       --data_dir=/vast/yl9315/data/Places365\
       --dataset ${INPUT_DATASET}\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --indices_dir ${INPUT_DATASET}/indices\
       --hparams "{\"lr\":1e-3, \"batch_size\":512, \"T\":9, \"tc_temperature\":9, \"index_dataset\":0, \"vit\":1, \"clip\":0, \"dinov2\":1, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --checkpoint_freq 500\
       --freeze\
       --seed $INPUT_SEED\
       --trial_seed $INPUT_SEED\
       --start_step 1\
       --steps 3001\
       --test_envs 2\
       --output_dir ${INPUT_DATASET}/ceiling/${INPUT_SEED}/${INPUT_SUBSET}_${INPUT_ST}\
       --deterministic_only
       
       #--checkpoint ${INPUT_DATASET}/st/0/${INPUT_SUBSET}_18_a3_1_${INPUT_ST}\
       #--checkpoint_erm\
       #--load_head\
       
       #--checkpoint ${INPUT_DATASET}/st/${SEED}/1_50_${INPUT_SUBSET}_${INPUT_ST}\
       #--checkpoint_erm\
       #--load_head
       #--deterministic_only\