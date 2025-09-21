#!/bin/bash
#SBATCH --job-name=toceiling_lp_dinov2
#SBATCH --array=6-7
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 16:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%A_%a.out            # error file name (same to watch just one file)


TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
module load anaconda3/2024.02
export HF_HOME=/scratch/jl4476/$TASK_ID
export TORCH_HOME=/scratch/jl4476/$TASK_ID
source $HOME/.bashrc
#export PATH=/scratch/yl9315/miniconda3/bin:$PATH
conda activate Domain

# Deterministic
export CUBLAS_WORKSPACE_CONFIG=:4096:8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Define your input file lists
TC=("dinov2")  # 3 teachers
#SUBSET=(0.1625 0.195 0.2275 0.24375 0.26 0.2925 0.325)
#SUBSET_INT=(4550 5460 6370 6825 7280 8190 9100)
SUBSET=(0.195)
SUBSET_INT=(5460)
SUBSET=(0.0643 0.09643 0.12857 0.25 0.4643)
SUBSET_INT=(1800 2700 3600 7000 13000)
SUBSET=(0.0143 0.03572 0.05715 0.07858 0.1 0.12143 0.143)
SUBSET_INT=(400 1000 1600 2200 2800 3400 4000)
#SUBSET_ST=(0.17143 0.25 0.3393 0.4643 0.57143)
#SUBSET_ST_INT=(4800 7000 9500 13000 16000)
# SUBSET_ST=(0.25 0.4643 0.57143 0.7143 1)
# SUBSET_ST_INT=(7000 13000 16000 20000 28000)
SUBSET=(0.1429 0.2858 0.4286 0.57143 0.7143 0.857143 1)
SUBSET_INT=(4000 8000 12000 16000 20000 24000 28000)
#SUBSET=(0.1 0.115 0.13 0.147 0.1625 0.195 0.2275)
#SUBSET_INT=(2800 3220 3640 4116 4550 5460 6370)
DATASET=("CIFAR10") 

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(16)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
IDX_SUBSET=$((TASK_ID / ${#DATASET[@]})) # Get index for ST
IDX_DATASET=$((TASK_ID % ${#DATASET[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_DATASET=${DATASET[$IDX_DATASET]}
INPUT_SUBSET=${SUBSET[$IDX_SUBSET]}
INPUT_SUBSET_INT=${SUBSET_INT[$IDX_SUBSET]}
INPUT_TC=${TC[0]}
SEED=0

mkdir -p /scratch/jl4476/${DATASET}/indices
mkdir -p /scratch/jl4476/data
rsync -av --ignore-existing ./domainbed/data/MNIST/ /scratch/$USER/data/
rsync -av --ignore-existing ./${DATASET}/indices/ /scratch/$USER/${DATASET}/indices/

python3 -m domainbed.scripts.ceiling_single\
       --data_dir=/scratch/$USER/data/\
       --dataset ${INPUT_DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":3e-3, \"batch_size\":512, \"input_shape\":224, \"resnet\":0, \"resnet50_augmix\":0, \"vit\":1, \"dinov2\":1, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"s14\"}"\
       --checkpoint_freq 500\
       --steps 4001\
       --start_step 1\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/ceiling_single/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_s14\
       --freeze\
       --mse\
       --deterministic_only\

python3 -m domainbed.scripts.ceiling_single\
       --data_dir=/scratch/$USER/data/\
       --dataset ${INPUT_DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":1e-4, \"batch_size\":512, \"input_shape\":224, \"resnet\":0, \"resnet50_augmix\":0, \"vit\":1, \"dinov2\":1, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"s14\"}"\
       --checkpoint_freq 1000\
       --steps 8001\
       --start_step 4001\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/ceiling_single/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_s14\
       --freeze\
       --mse\
       --deterministic_only\
       --checkpoint /scratch/jl4476/${INPUT_DATASET}/ceiling_single/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_s14\
       --checkpoint_erm\
       --load_head\

python3 -m domainbed.scripts.ceiling_single\
       --data_dir=/scratch/$USER/data/\
       --dataset ${INPUT_DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":3e-5, \"batch_size\":512, \"input_shape\":224, \"resnet\":0, \"resnet50_augmix\":0, \"vit\":1, \"dinov2\":1, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"s14\"}"\
       --checkpoint_freq 1000\
       --steps 16001\
       --start_step 8001\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/ceiling_single/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_s14\
       --freeze\
       --mse\
       --deterministic_only\
       --checkpoint /scratch/jl4476/${INPUT_DATASET}/ceiling_single/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_s14\
       --checkpoint_erm\
       --load_head\

python3 -m domainbed.scripts.eval_mse\
       --data_dir=/scratch/$USER/data/\
       --dataset ${INPUT_DATASET}\
       --test_env 2\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/ceiling_single/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_s14\
       --checkpoint /scratch/jl4476/${INPUT_DATASET}/ceiling_single/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_s14\
       --hparams "{\"lr\":1e-4, \"batch_size\":512, \"input_shape\":224, \"T\":9, \"tc_temperature\":9, \"index_dataset\":1, \"model\": \"${INPUT_TC}\", \"resnet\":0, \"resnet50_augmix\":0, \"vit\":1, \"clip\":0, \"dinov2\":1, \"dino\":0, \"arch\": \"s14\"}"\
