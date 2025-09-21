#!/bin/bash
#SBATCH --job-name=eval_mse_tc
#SBATCH --array=1-48
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --time 00:15:00              # maximum execution time (HH:MM:SS)
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
SEED=(0 1 2)
TC=("vit_tiny" "vit_tiny" "vit_tiny" "vit_tiny" "vit_tiny" "vit_tiny" "vit_base" "vit_base" "vit_base" "vit_base" "vit_base" "resnet18" "resnet18" "resnet18" "resnet18" "resnet18")
SUBSET_INT=(30 70 140 200 400 600 70 200 400 600 1000 200 600 1000 1400 1800)

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
INPUT_TC=${TC[$IDX_TC]}
DATASET=${DATASETS[0]}
LR=${LRS[0]}

mkdir -p /scratch/jl4476/data
rsync -av --ignore-existing ./domainbed/data/MNIST/ /scratch/$USER/data/

python3 -m domainbed.scripts.eval_mse_by_dict\
       --data_dir=/scratch/jl4476/data/\
       --algorithm StudentSingle\
       --dataset ${DATASET}\
       --dc_dir ./${DATASET}/diction/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}/diction.json\
       --hparams "{\"lr\":3e-5, \"batch_size\":512,\"T\":9, \"tc_temperature\":9, \"index_dataset\":1, \"input_shape\":224, \"resnet18\":1, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"18\"}"\
       --test_env 1\
       
       #--output_dir tc_${DATASET}/0/${INPUT_SUBSET_INT}_${INPUT_TC}\
       #--checkpoint ${DATASET}/tc/0/${INPUT_SUBSET_INT}_${INPUT_TC}\