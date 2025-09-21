#!/bin/bash
#SBATCH --job-name=tc_resnet18
#SBATCH --array=1-1
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 6:00:00              # maximum execution time (HH:MM:SS)
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Define your input file lists
TC=("resnet18")  # 3 teachers
#SUBSET=(0.00273 0.0036 0.005 0.01 0.015 0.02 0.02322)
#SUBSET_INT=(75 100 140 280 420 560 650)
#SUBSET=(0.00714 0.02143 0.03571 0.05 0.0643)
#SUBSET_INT=(200 600 1000 1400 1800)
SUBSET=(0.001)
SUBSET_INT=(28)
#SUBSET=(0.5 0.25 0.125)  # 3 students
DATASET=("ColoredMNISTID") # 3 datasets

# Calculate the total number of combinations
# TOTAL_COMBINATIONS=$(24)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_SUBSET=$((TASK_ID / ${#DATASET[@]})) # Get index for ST
IDX_DATASET=$((TASK_ID % ${#DATASET[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_DATASET=${DATASET[$IDX_DATASET]}
INPUT_SUBSET=${SUBSET[$IDX_SUBSET]}
INPUT_SUBSET_INT=${SUBSET_INT[$IDX_SUBSET]}
INPUT_TC=${TC[0]}
SEED=0

mkdir -p /scratch/jl4476/${INPUT_DATASET}/indices
mkdir -p /scratch/jl4476/data
rsync -av --ignore-existing ./domainbed/data/MNIST/ /scratch/$USER/data/
rsync -av --ignore-existing ./${INPUT_DATASET}/indices/ /scratch/$USER/${INPUT_DATASET}/indices/

python3 -m domainbed.scripts.tc\
       --data_dir=/scratch/jl4476/data/\
       --dataset ${INPUT_DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":3e-3, \"batch_size\":512, \"input_shape\":224, \"resnet18\":1, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"18\"}"\
       --checkpoint_freq 1000\
       --steps 10001\
       --start_step 1\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --freeze\
       --deterministic_only\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/tc/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}\

python3 -m domainbed.scripts.tc\
       --data_dir=/scratch/jl4476/data/\
       --dataset ${INPUT_DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":1e-4, \"batch_size\":512, \"input_shape\":224, \"resnet18\":1, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"18\"}"\
       --checkpoint_freq 1000\
       --steps 30001\
       --start_step 10001\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --freeze\
       --deterministic_only\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/tc/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}\
       --checkpoint /scratch/jl4476/${INPUT_DATASET}/tc/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}\
       --checkpoint_erm\

python3 -m domainbed.scripts.tc\
       --data_dir=/scratch/jl4476/data/\
       --dataset ${INPUT_DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":3e-5, \"batch_size\":512, \"input_shape\":224, \"resnet18\":1, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"18\"}"\
       --checkpoint_freq 1000\
       --steps 40001\
       --start_step 30001\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --freeze\
       --deterministic_only\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/tc/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}\
       --checkpoint /scratch/jl4476/${INPUT_DATASET}/tc/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}\
       --checkpoint_erm\

python3 -m domainbed.scripts.diction_single\
       --data_dir=/scratch/jl4476/data/\
       --teacher_dir /scratch/jl4476/${INPUT_DATASET}/tc/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}\
       --hparams "{\"lr\":3e-3, \"batch_size\":512, \"input_shape\":224, \"resnet18\":1, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"18\"}"\
       --dataset ${INPUT_DATASET}\
       --test_env 1\
       --seed $SEED\
       --trial_seed $SEED\
       --teacher_name model.pkl\
       --deterministic_only\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/diction/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}\
       
mkdir -p ${INPUT_DATASET}/diction/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}/
cp /scratch/$USER/${INPUT_DATASET}/diction/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}/diction.json ./${INPUT_DATASET}/diction/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}/
       