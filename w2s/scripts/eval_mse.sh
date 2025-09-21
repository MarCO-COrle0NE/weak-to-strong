#!/bin/bash
#SBATCH --job-name=eval_mse_tc
#SBATCH --array=6
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 1:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%A_%a.out            # error file name (same to watch just one file)

module purge
source $HOME/.bashrc
export PATH=/scratch/yl9315/miniconda3/bin:$PATH
module load cuda/11.6.2
conda activate DomainBed

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Define your input file lists
TC=("resnet18")  # 3 teachers
STS="resnet50"
ST=("50")
DATASETS=("ColoredMNISTID") 
LRS=(3e-3)
#SUBSET_ST=(0.1625 0.2275 0.24375 0.26 0.2925 0.325)
#SUBSET_ST_INT=(4550 6370 6825 7280 8190 9100)
#SUBSET=(0.01)
#SUBSET_INT=(280)
SUBSET_ST=(0.195)
SUBSET_ST_INT=(5460)
SUBSET=(0.00273 0.0036 0.005 0.01 0.015 0.02 0.02322)
SUBSET_INT=(75 100 140 280 420 560 650)
SEED=(0)

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(160)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_TC=$((TASK_ID / $((${#SUBSET_ST[@]} * ${#SEED[@]})))) # Get index for ST
IDX_ST=$(($((TASK_ID % $((${#SUBSET_ST[@]} * ${#SEED[@]})))) / ${#SEED[@]})) # Get index for TC
IDX_SEED=$((TASK_ID % ${#SEED[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_SEED=${SEED[$IDX_SEED]}
INPUT_SUBSET_ST=${SUBSET_ST[$IDX_ST]}
INPUT_SUBSET=${SUBSET[$IDX_TC]}
INPUT_SUBSET_ST_INT=${SUBSET_ST_INT[$IDX_ST]}
INPUT_SUBSET_INT=${SUBSET_INT[$IDX_TC]}
INPUT_ST=${ST[0]}
INPUT_TC=${TC[0]}
DATASET=${DATASETS[0]}
LR=${LRS[0]}
       

python3 -m domainbed.scripts.eval_mse\
       --data_dir=/vast/yl9315/data/Places365\
       --dataset ${DATASET}\
       --hparams "{\"lr\":3e-5, \"batch_size\":512, \"input_shape\":224, \"resnet18\":1, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"18\"}"\
       --test_env 1\
       --output_dir tc_${DATASET}/0/${INPUT_SUBSET_INT}_${INPUT_TC}\
       --checkpoint ${DATASET}/tc/0/${INPUT_SUBSET_INT}_${INPUT_TC}\