#!/bin/bash
#SBATCH --job-name=ceiling_resnet_mse
#SBATCH --array=1,2,6,7
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 3:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%A_%a.out            # error file name (same to watch just one file)

module load anaconda3/2024.02
export HF_HOME=/scratch/jl4476
export TORCH_HOME=/scratch/jl4476
source $HOME/.bashrc
#export PATH=/scratch/yl9315/miniconda3/bin:$PATH
conda activate Domain

# Deterministic
#export CUBLAS_WORKSPACE_CONFIG=:4096:8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Define your input file lists
TC=("mobile")  # 3 teachers
STS="resnet"
ST=("18" "34" "50" "101" "152")
DATASETS=("ColoredMNISTMixed" "CIFAR10") 
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
LR=${LRS[0]}
       
mkdir -p /scratch/jl4476/data
rsync -av --ignore-existing ./domainbed/data/MNIST/ /scratch/$USER/data/

python3 -m domainbed.scripts.ceiling\
       --data_dir=/scratch/$USER/data/\
       --dataset ${DATASET}\
       --hparams "{\"lr\":$LR, \"batch_size\":256, \"input_shape\":224, \"model\": \"${STS}\", \"resnet\":1, \"vit\":0, \"resnet50_augmix\":0, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --seed $INPUT_SEED\
       --trial_seed $INPUT_SEED\
       --start_step 1\
       --steps 6001\
       --checkpoint_freq 500\
       --test_envs 2\
       --output_dir /scratch/jl4476/${DATASET}/ceiling/${INPUT_SEED}/${STS}_${INPUT_ST}\
       --freeze\
       --mse\

python3 -m domainbed.scripts.ceiling\
       --data_dir=/scratch/$USER/data/\
       --dataset ${DATASET}\
       --hparams "{\"lr\":1e-4, \"batch_size\":256, \"input_shape\":224, \"model\": \"${STS}\", \"resnet\":1, \"vit\":0, \"resnet50_augmix\":0, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --seed $INPUT_SEED\
       --trial_seed $INPUT_SEED\
       --start_step 6001\
       --steps 8001\
       --checkpoint_freq 500\
       --test_envs 2\
       --output_dir /scratch/jl4476/${DATASET}/ceiling/${INPUT_SEED}/${STS}_${INPUT_ST}\
       --checkpoint /scratch/jl4476/${DATASET}/ceiling/${INPUT_SEED}/${STS}_${INPUT_ST}\
       --checkpoint_erm\
       --freeze\
       --mse\

python3 -m domainbed.scripts.ceiling\
       --data_dir=/scratch/$USER/data/\
       --dataset ${DATASET}\
       --hparams "{\"lr\":3e-5, \"batch_size\":256, \"input_shape\":224, \"model\": \"${STS}\", \"resnet\":1, \"vit\":0, \"resnet50_augmix\":0, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --seed $INPUT_SEED\
       --trial_seed $INPUT_SEED\
       --start_step 8001\
       --steps 10001\
       --checkpoint_freq 500\
       --test_envs 2\
       --output_dir /scratch/jl4476/${DATASET}/ceiling/${INPUT_SEED}/${STS}_${INPUT_ST}\
       --checkpoint /scratch/jl4476/${DATASET}/ceiling/${INPUT_SEED}/${STS}_${INPUT_ST}\
       --checkpoint_erm\
       --freeze\
       --mse\
       
python3 -m domainbed.scripts.eval_mse\
       --data_dir=/scratch/$USER/data/\
       --dataset ${DATASET}\
       --test_env 2\
       --output_dir /scratch/jl4476/${DATASET}/ceiling/${INPUT_SEED}/${STS}_${INPUT_ST}\
       --checkpoint /scratch/jl4476/${DATASET}/ceiling/${INPUT_SEED}/${STS}_${INPUT_ST}\
       --hparams "{\"lr\":1e-4, \"batch_size\":512, \"input_shape\":224, \"T\":9, \"tc_temperature\":9, \"index_dataset\":1, \"model\": \"${STS}\", \"resnet\":1, \"resnet50_augmix\":0, \"vit\":0, \"clip\":0, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\

       #--deterministic_only
       #--checkpoint ${DATASET}/st_mamba/${INPUT_SEED}/${INPUT_SUBSET}_${INPUT_TC}_${INPUT_SUBSET_ST}_${INPUT_ST}\
       #--checkpoint_algorithm StudentSingle\