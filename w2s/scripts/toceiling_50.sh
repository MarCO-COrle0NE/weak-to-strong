#!/bin/bash
#SBATCH --job-name=toceiling_lp_resnet50_single
#SBATCH --array=1-4
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=56GB
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 6:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%A_%a.out            # error file name (same to watch just one file)

module load anaconda3/2024.02
export HF_HOME=/scratch/jl4476
export TORCH_HOME=/scratch/jl4476
source $HOME/.bashrc
#export PATH=/scratch/yl9315/miniconda3/bin:$PATH
conda activate Domain

# Deterministic
export CUBLAS_WORKSPACE_CONFIG=:4096:8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Define your input file lists
TC=("resnet50")  # 3 teachers
#SUBSET=(0.1625 0.195 0.2275 0.24375 0.26 0.2925 0.325)
#SUBSET_INT=(4550 5460 6370 6825 7280 8190 9100)
SUBSET=(0.195)
SUBSET_INT=(5460)
SUBSET=(0.1 0.115 0.13 0.147 0.1625 0.195 0.2275 0.24375 0.26 0.2925 0.325)
SUBSET_INT=(2800 3220 3640 4116 4550 5460 6370 6825 7280 8190 9100)
#SUBSET=(0.1 0.115 0.13 0.147 0.1625 0.195 0.2275)
#SUBSET_INT=(2800 3220 3640 4116 4550 5460 6370)
SUBSET=(0.4286 0.5357143 0.642857143 0.75 0.857143 1)
SUBSET_INT=(12000 15000 18000 21000 24000 28000)
SUBSET=(0.375 0.482143 0.5893 0.69643)
SUBSET_INT=(10500 13500 16500 19500)
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
INPUT_SUBSET_INT=${SUBSET_INT[$IDX_SUBSET]}
INPUT_TC=${TC[0]}
SEED=1

mkdir -p /scratch/jl4476/${DATASET}/indices
mkdir -p /scratch/jl4476/data
rsync -av --ignore-existing ./domainbed/data/MNIST/ /scratch/$USER/data/
rsync -av --ignore-existing ./${DATASET}/indices/ /scratch/$USER/${DATASET}/indices/

python3 -m domainbed.scripts.ceiling_single\
       --data_dir=/scratch/$USER/data/\
       --dataset ${INPUT_DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":3e-3, \"batch_size\":512, \"input_shape\":224, \"resnet\":0, \"resnet50_augmix\":1, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"50\"}"\
       --checkpoint_freq 1000\
       --steps 12001\
       --start_step 1\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/ceiling_single/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}\
       --freeze\
       --mse\
       --deterministic_only\

python3 -m domainbed.scripts.ceiling_single\
       --data_dir=/scratch/$USER/data/\
       --dataset ${INPUT_DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":1e-4, \"batch_size\":512, \"input_shape\":224, \"resnet\":0, \"resnet50_augmix\":1, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"50\"}"\
       --checkpoint_freq 1000\
       --steps 18001\
       --start_step 12001\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/ceiling_single/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}\
       --freeze\
       --mse\
       --deterministic_only\
       --checkpoint /scratch/jl4476/${INPUT_DATASET}/ceiling_single/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}\
       --load_head\

python3 -m domainbed.scripts.ceiling_single\
       --data_dir=/scratch/$USER/data/\
       --dataset ${INPUT_DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":1e-4, \"batch_size\":512, \"input_shape\":224, \"resnet\":0, \"resnet50_augmix\":1, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"50\"}"\
       --checkpoint_freq 1000\
       --steps 20001\
       --start_step 18001\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/ceiling_single/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}\
       --freeze\
       --mse\
       --deterministic_only\
       --checkpoint /scratch/jl4476/${INPUT_DATASET}/ceiling_single/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}\
       --load_head\