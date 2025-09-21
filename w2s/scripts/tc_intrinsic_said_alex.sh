#!/bin/bash
#SBATCH --job-name=tc_intrinsic_said_alex
#SBATCH --array=1-3
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
conda activate DomainBed

# Deterministic
#export CUBLAS_WORKSPACE_CONFIG=:4096:8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Define your input file lists
TC=("alexnet")  # 3 teachers
#SUBSET=(1 0.02 0.05 0.07 0.1 0.2 0.5 0.75 1)
SUBSET=(1)
INTRINSIC_DIM=(100 200 300 400 500 600 700 800)
#SUBSET=(0.5 0.25 0.125)  # 3 students
DATASET=("ColoredMNISTID") 

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(16)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_DIM=$((TASK_ID / ${#TC[@]})) # Get index for ST
IDX_TC=$((TASK_ID % ${#TC[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_DATASET=${DATASET[0]}
INPUT_DIM=${INTRINSIC_DIM[$IDX_DIM]}
INPUT_SUBSET=${SUBSET[0]}
INPUT_TC=${TC[$IDX_TC]}

python3 -m domainbed.scripts.tc\
       --data_dir=/vast/yl9315/data/Places365\
       --dataset ${INPUT_DATASET}\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":1e-1, \"SGD\":1,\"batch_size\":512, \"input_shape\":224, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"${INPUT_TC}\", \"${INPUT_TC}\":1,  \"mamba\":0, \"arch\": \"${INPUT_TC}\"}"\
       --checkpoint_freq 2000\
       --steps 40001\
       --start_step 20001\
       --test_envs 2\
       --intrinsic_dimension $INPUT_DIM\
       --said\
       --output_dir ${INPUT_DATASET}/tc_intrinsic_said/${INPUT_SUBSET}_${INPUT_TC}_${INPUT_DIM}\
       --checkpoint ${INPUT_DATASET}/tc_intrinsic_said/${INPUT_SUBSET}_${INPUT_TC}_${INPUT_DIM}\
       --checkpoint_erm\
       
       #--deterministic_only
       
       #--indices_dir ${INPUT_DATASET}/indices\