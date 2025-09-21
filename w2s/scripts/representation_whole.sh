#!/bin/bash
#SBATCH --job-name=representation_whole
#SBATCH --array=1-4
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --nodelist=neu301
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 1:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%A_%a.out            # error file name (same to watch just one file)

module load anaconda3/2024.02
export HF_HOME=/scratch/jl4476
export TORCH_HOME=/scratch/jl4476
source $HOME/.bashrc
#export PATH=/scratch/yl9315/miniconda3/bin:$PATH
conda activate Domain

# Deterministic
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Define your input file lists
TC=("resnet" "resnet" "resnet-d" "resnet-d")  # 3 teachers
#ARCHS=("50" "101" "18" "34")
FLIP=(0 0.1 0.2 0.3)
#SUBSET=(1 0.02 0.05 0.07 0.1 0.2 0.5 0.75 1)
SUBSET=(34)
#SUBSET=(0.5 0.25 0.125)  # 3 students
DATASET=("ColoredMNISTMixedNew")  #neu301, neu304
#DATASET=("ColoredMNISTMixedNew")

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(16)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_ARCH=$((TASK_ID / ${#DATASET[@]})) # Get index for ST
IDX_DATASET=$((TASK_ID % ${#DATASET[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_DATASET=${DATASET[$IDX_DATASET]}
INPUT_ARCH=${ARCHS[$IDX_ARCH]}
INPUT_FLIP=${FLIP[$IDX_ARCH]}
INPUT_TC=${TC[$IDX_ARCH]}

mkdir -p /scratch/jl4476/data
rsync -av --ignore-existing ./domainbed/data/MNIST/ /scratch/$USER/data/

# python3 -m domainbed.scripts.representation_whole_clip\
#      --data_dir=/scratch/jl4476/data\
#      --dataset ${INPUT_DATASET}\
#      --hparams "{\"batch_size\":512, \"input_shape\":224, \"mamba\":0, \"resnet\":0, \"resnet50_augmix\":0, \"model\": \"mamba\", \"vit\":0, \"arch\": \"${INPUT_SUBSET}\", \"dino\":0, \"dinov2\":0, \"clip\":1}"\
#      --test_envs 2\
#      --output_dir /scratch/jl4476/${INPUT_DATASET}/\

# python3 -m domainbed.scripts.representation_whole_vit\
#      --data_dir=/scratch/jl4476/data\
#      --dataset ${INPUT_DATASET}\
#      --hparams "{\"batch_size\":512, \"input_shape\":224, \"resnet\":0, \"resnet50_augmix\":0, \"model\": \"resnet\", \"vit\":1, \"arch\": \"${INPUT_SUBSET}\", \"dino\":0, \"dinov2\":0, \"clip\":0}"\
#      --test_envs 2\
#      --output_dir /scratch/jl4476/${INPUT_DATASET}/\

python3 -m domainbed.scripts.representation_whole\
     --data_dir=/scratch/jl4476/data\
     --dataset ${INPUT_DATASET}\
     --hparams "{\"batch_size\":512, \"input_shape\":224, \"flip_prob\":${INPUT_FLIP}, \"resnet\":1, \"resnet50_augmix\":0, \"model\": \"resnet\", \"vit\":0, \"arch\": \"${INPUT_SUBSET}\", \"dino\":0, \"dinov2\":0, \"clip\":0}"\
     --test_envs 2\
     --output_dir /scratch/jl4476/${INPUT_DATASET}_${INPUT_FLIP}/\

# mkdir -p tc_${INPUT_DATASET}/resnet_${INPUT_SUBSET}_pre
# cp /scratch/jl4476/tc_${INPUT_DATASET}/resnet_${INPUT_SUBSET}_pre/representations.npz tc_${INPUT_DATASET}/resnet_${INPUT_SUBSET}_pre/

# python3 -m domainbed.scripts.representation\
#      --data_dir=/scratch/jl4476/data\
#      --dataset ${INPUT_DATASET}\
#      --hparams "{\"batch_size\":512, \"input_shape\":224, \"resnet18\":1, \"resnet50_augmix\":0, \"model\": \"mamba\", \"vit\":0, \"arch\": \"base\", \"dino\":0, \"dinov2\":0, \"clip\":0}"\
#      --test_envs 2\
#      --whole_env\
#      --output_dir /scratch/jl4476/tc_${INPUT_DATASET}/resnet_18_pre\

# mkdir -p tc_${INPUT_DATASET}/resnet_18_pre
# cp /scratch/jl4476/tc_${INPUT_DATASET}/resnet_18_pre/representations.npz tc_${INPUT_DATASET}/resnet_18_pre/

# python3 -m domainbed.scripts.representation\
#       --data_dir=/scratch/jl4476/data\
#       --dataset ${INPUT_DATASET}\
#       --hparams "{\"batch_size\":512, \"input_shape\":224, \"dinov2\":1, \"resnet50_augmix\":0, \"vit\":1, \"model\": \"dinov2\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"s14\"}"\
#       --test_envs 2\
#       --whole_env\
#       --output_dir /scratch/jl4476/tc_${INPUT_DATASET}/dinov2_s14_pre\

# mkdir -p tc_${INPUT_DATASET}/dinov2_s14_pre
# cp /scratch/jl4476/tc_${INPUT_DATASET}/dinov2_s14_pre/representations.npz tc_${INPUT_DATASET}/dinov2_s14_pre/

# python3 -m domainbed.scripts.representation\
#       --data_dir=/scratch/jl4476/data\
#       --dataset ${INPUT_DATASET}\
#       --hparams "{\"batch_size\":512, \"input_shape\":224, \"dinov2\":1, \"resnet50_augmix\":0, \"vit\":1, \"model\": \"dinov2\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"b14\"}"\
#       --test_envs 2\
#       --whole_env\
#       --output_dir /scratch/jl4476/tc_${INPUT_DATASET}/dinov2_b14_pre\

# mkdir -p tc_${INPUT_DATASET}/dinov2_b14_pre
# cp /scratch/jl4476/tc_${INPUT_DATASET}/dinov2_b14_pre/representations.npz tc_${INPUT_DATASET}/dinov2_b14_pre/


# python3 -m domainbed.scripts.representation\
#       --data_dir=/scratch/jl4476/data\
#       --dataset ${INPUT_DATASET}\
#       --hparams "{\"batch_size\":512, \"input_shape\":224, \"dinov2\":0, \"resnet50_augmix\":0, \"vit\":1, \"model\": \"vit\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"tiny\", \"dino\":0, \"dinov2\":0, \"clip\":0}"\
#       --test_envs 2\
#       --whole_env\
#       --output_dir /scratch/jl4476/tc_${INPUT_DATASET}/vit_tiny_pre\

# mkdir -p tc_${INPUT_DATASET}/vit_tiny_pre
# cp /scratch/jl4476/tc_${INPUT_DATASET}/vit_tiny_pre/representations.npz tc_${INPUT_DATASET}/vit_tiny_pre/

# python3 -m domainbed.scripts.representation\
#       --data_dir=/scratch/jl4476/data\
#       --dataset ${INPUT_DATASET}\
#       --hparams "{\"batch_size\":512, \"input_shape\":224, \"dinov2\":0, \"resnet50_augmix\":0, \"vit\":1, \"model\": \"vit\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"small\", \"dino\":0, \"dinov2\":0, \"clip\":0}"\
#       --test_envs 2\
#       --whole_env\
#       --output_dir /scratch/jl4476/tc_${INPUT_DATASET}/vit_small_pre\

# mkdir -p tc_${INPUT_DATASET}/vit_small_pre
# cp /scratch/jl4476/tc_${INPUT_DATASET}/vit_small_pre/representations.npz tc_${INPUT_DATASET}/vit_small_pre/


# python3 -m domainbed.scripts.representation\
#      --data_dir=/scratch/jl4476/data\
#      --dataset ${INPUT_DATASET}\
#      --hparams "{\"batch_size\":512, \"input_shape\":224, \"mamba\":1, \"resnet50_augmix\":0, \"model\": \"mamba\", \"vit\":0, \"arch\": \"base\", \"dino\":0, \"dinov2\":0, \"clip\":0}"\
#      --test_envs 2\
#      --whole_env\
#      --output_dir /scratch/jl4476/tc_${INPUT_DATASET}/mamba_base_pre\

# mkdir -p tc_${INPUT_DATASET}/mamba_tiny_pre
# cp /scratch/jl4476/tc_${INPUT_DATASET}/mamba_base_pre/representations.npz tc_${INPUT_DATASET}/mamba_base_pre/

       
# python3 -m domainbed.scripts.representation\
#       --data_dir=/scratch/jl4476/data\
#       --dataset ${INPUT_DATASET}\
#       --hparams "{\"batch_size\":64, \"input_shape\":224, \"resnet18\":0, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"vit_large\", \"vit\":1, \"arch\": \"large\", \"dino\":0, \"dinov2\":0, \"clip\":0}"\
#       --test_envs 2\
#       --whole_env\
#       --output_dir /scratch/jl4476/tc_${INPUT_DATASET}/vit_large_pre\

# mkdir -p tc_${INPUT_DATASET}/vit_large_pre
# cp /scratch/jl4476/tc_${INPUT_DATASET}/vit_large_pre/representations.npz tc_${INPUT_DATASET}/vit_large_pre/


#python3 -m domainbed.scripts.representation\
#       --data_dir=/scratch/jl4476/data\
#       --dataset ${INPUT_DATASET}\
#       --hparams "{\"batch_size\":512, \"input_shape\":224, \"resnet18\":0, \"resnet50_augmix\":0, \"vit\":1, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"${INPUT_TC}\"}"\
#       --test_envs 1\
#       --output_dir /scratch/jl4476/tc_${INPUT_DATASET}/${INPUT_TC}_pre\