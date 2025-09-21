#!/bin/bash
#SBATCH --job-name=tost_mobile
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
STS="mobile"
ST=("mobile")
#ST=("wee" "pwee" "little" "medium" "betwixt")
DATASETS=("ColoredMNISTID") 
LRS=(3e-3)
SUBSET_ST=(1)
SUBSET=(1)
SEED=(0)

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(160)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_DATASET=$((TASK_ID / $((${#ST[@]} * ${#SEED[@]})))) # Get index for ST
IDX_ST=$(($((TASK_ID % $((${#ST[@]} * ${#SEED[@]})))) / ${#SEED[@]})) # Get index for TC
IDX_SEED=$((TASK_ID % ${#SEED[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_SEED=${SEED[$IDX_SEED]}
INPUT_ST=${ST[$IDX_ST]}
INPUT_SUBSET=${SUBSET[0]}
INPUT_SUBSET_ST=${SUBSET_ST[0]}
INPUT_TC=${TC[0]}
DATASET=${DATASETS[$IDX_DATASET]}
LR=${LRS[$IDX_DATASET]}
       
python3 -m domainbed.scripts.st\
       --data_dir=/vast/yl9315/data/Places365\
       --dc_dir ${DATASET}/diction/${INPUT_SEED}/${INPUT_SUBSET}_${INPUT_TC}/diction.json\
       --algorithm StudentSingle\
       --dataset ${DATASET}\
       --task domain_adaptation\
       --hparams "{\"lr\":$LR, \"batch_size\":512, \"input_shape\":224, \"T\":9, \"tc_temperature\":9, \"index_dataset\":1, \"model\": \"${STS}\", \"mobile\":1, \"vit\":0, \"clip\":0, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --dinov\
       --seed $INPUT_SEED\
       --trial_seed $INPUT_SEED\
       --subset_holdout_fraction ${INPUT_SUBSET_ST}\
       --start_step 1\
       --steps 15001\
       --checkpoint_freq 1000\
       --test_envs 2\
       --output_dir ${DATASET}/st_mobile/${INPUT_SEED}/${INPUT_SUBSET}_${INPUT_TC}_${INPUT_SUBSET_ST}_${INPUT_ST}\
       --freeze\
       --deterministic_only