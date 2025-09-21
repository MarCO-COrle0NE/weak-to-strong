#!/bin/bash
#SBATCH --job-name=tc_ft_resnet
#SBATCH --array=1-1
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
TC=("resnet18")  # 3 teachers
#SUBSET=(1 0.02 0.05 0.07 0.1 0.2 0.5 0.75 1)
SUBSET=(1)
#SUBSET=(0.5 0.25 0.125)  # 3 students
DATASET=("ColoredMNISTID") 

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(16)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_SUBSET=$((TASK_ID / ${#DATASET[@]})) # Get index for ST
IDX_DATASET=$((TASK_ID % ${#DATASET[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_DATASET=${DATASET[$IDX_DATASET]}
INPUT_SUBSET=${SUBSET[$IDX_SUBSET]}
INPUT_TC=${TC[0]}

python3 -m domainbed.scripts.tc\
       --data_dir=/vast/yl9315/data/Places365\
       --dataset ${INPUT_DATASET}\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":5e-5, \"batch_size\":512, \"input_shape\":224, \"resnet18\":1, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"${INPUT_TC}\"}"\
       --checkpoint_freq 1000\
       --checkpoint_erm\
       --steps 30001\
       --start_step 20001\
       --test_envs 2\
       --output_dir ${INPUT_DATASET}/tc_ft/${INPUT_SUBSET}_${INPUT_TC}\
       --save_model_every_checkpoint\
       --deterministic_only\
       --checkpoint ${INPUT_DATASET}/tc_ft/${INPUT_SUBSET}_${INPUT_TC}
       
       #--indices_dir ${INPUT_DATASET}/indices\