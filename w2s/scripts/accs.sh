#!/bin/bash
#SBATCH --job-name=accs_st
#SBATCH --array=1-1
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
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

SEED=(0)
#STL=("vit" "mamba" "dino" "dinov2")
#STL=("vit" "mamba" "dinov2" "clip")
STL=("vit" "mamba" "dinov2" "sbb" "resnet-d")
DATASET="Places365" # 3 datasets

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(40)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_ST=$((TASK_ID / ${#SEED[@]})) # Get index for ST
IDX_SEED=$((TASK_ID % ${#SEED[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_SEED=${SEED[$IDX_SEED]}
ST=${STL[$IDX_ST]}

python3 -m domainbed.scripts.accs --output_dir st_ColoredMNISTID/0/18_50/ --seed 0\

#python3 -m domainbed.scripts.accs_vit --output_dir st/${INPUT_SEED}/18_a3_${ST}/CIFAR10 --seed $INPUT_SEED --dataset="CIFAR10" --st=${ST}\

#python3 -m domainbed.scripts.accs_vit --output_dir st/${INPUT_SEED}/18_a3_${ST}/CIFAR100 --seed $INPUT_SEED --dataset="CIFAR100" --st=${ST}\

#python3 -m domainbed.scripts.accs_vit --output_dir st/${INPUT_SEED}/18_a3_${ST}/Places365 --seed $INPUT_SEED --dataset="Places365" --st=${ST}