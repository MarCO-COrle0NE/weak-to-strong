#!/bin/bash
#SBATCH --job-name=diction_subset_resnet18
#SBATCH --array=1-4
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 2:00:00              # maximum execution time (HH:MM:SS)
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
SUBSET=(0.00273 0.0036 0.005 0.01 0.015 0.02 0.02322)
SUBSET_INT=(75 100 140 280 420 560 650)
DATASET=("ColoredMNISTID") # 3 datasets

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(24)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_SUBSET=$((TASK_ID / ${#DATASET[@]})) # Get index for ST
IDX_DATASET=$((TASK_ID % ${#DATASET[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_DATASET=${DATASET[$IDX_DATASET]}
INPUT_SUBSET=${SUBSET[$IDX_SUBSET]}
INPUT_SUBSET_INT=${SUBSET_INT[$IDX_SUBSET]}
INPUT_TC=${TC[0]}
SEED=2

python3 -m domainbed.scripts.diction_single\
       --data_dir=/vast/yl9315/data/Places365\
       --teacher_dir ${INPUT_DATASET}/tc/0/${INPUT_SUBSET_INT}_${INPUT_TC}\
       --hparams "{\"lr\":1e-4, \"batch_size\":512, \"input_shape\":224, \"resnet18\":1, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"18\"}"\
       --dataset ${INPUT_DATASET}\
       --test_env 1\
       --seed $SEED\
       --trial_seed $SEED\
       --teacher_name 'model.pkl'\
       --output_dir ${INPUT_DATASET}/diction/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}\
