#!/bin/bash
#SBATCH --job-name=tost_dinov2_hard
#SBATCH --array=1-49
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 16:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%A_%a.out            # error file name (same to watch just one file)

module load anaconda3/2024.02
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
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
TC=("resnet_50" "resnet_101" "resnet-d_18" "resnet-d_34")  # 3 teachers
ARCHS=("50" "101" "18" "34")
STS="dinov2"
ST=("s14")
DATASETS=("CIFAR10") 
LRS=(3e-3)
# SUBSET_ST=(0.25 0.4643 0.57143 0.7143 1)
# SUBSET_ST_INT=(7000 13000 16000 20000 28000)
# SUBSET=(0.00714 0.02143 0.03571 0.05 0.0643)
# SUBSET_INT=(200 600 1000 1400 1800)
SUBSET=(0.0143 0.03572 0.05715 0.07858 0.1 0.12143 0.143)
SUBSET_INT=(400 1000 1600 2200 2800 3400 4000)
SUBSET_ST=(0.1429 0.2858 0.4286 0.57143 0.7143 0.857143 1)
SUBSET_ST_INT=(4000 8000 12000 16000 20000 24000 28000)
SEED=(0)

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(160)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
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
       
mkdir -p /scratch/jl4476/${DATASET}/indices
mkdir -p /scratch/jl4476/data
mkdir -p /scratch/jl4476/${DATASET}/diction/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}/
rsync -av --ignore-existing ./domainbed/data/MNIST/ /scratch/$USER/data/
rsync -av --ignore-existing ./${DATASET}/diction/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}/ /scratch/$USER/${DATASET}/diction/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}/
rsync -av --ignore-existing ./${DATASET}/indices/ /scratch/$USER/${DATASET}/indices/

python3 -m domainbed.scripts.st\
       --data_dir=/scratch/$USER/data/\
       --dc_dir /scratch/jl4476/${DATASET}/diction/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}/diction.json\
       --indices_dir /scratch/jl4476/${DATASET}/indices\
       --algorithm StudentSingleHard\
       --dataset ${DATASET}\
       --task domain_adaptation\
       --hparams "{\"lr\":1e-3, \"batch_size\":512, \"input_shape\":224, \"T\":9, \"tc_temperature\":9, \"index_dataset\":1, \"model\": \"${STS}\", \"resnet50_augmix\":0, \"vit\":1, \"clip\":0, \"dinov2\":1, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --dinov\
       --seed $INPUT_SEED\
       --trial_seed $INPUT_SEED\
       --subset_holdout_fraction ${INPUT_SUBSET_ST}\
       --start_step 1\
       --steps 10001\
       --checkpoint_freq 500\
       --test_envs 2\
       --output_dir /scratch/jl4476/${DATASET}/st_dinov2/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
       --freeze\
       --mse\
       --deterministic_only\
       
python3 -m domainbed.scripts.st\
       --data_dir=/scratch/$USER/data/\
       --dc_dir /scratch/jl4476/${DATASET}/diction/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}/diction.json\
       --indices_dir /scratch/jl4476/${DATASET}/indices\
       --algorithm StudentSingleHard\
       --dataset ${DATASET}\
       --task domain_adaptation\
       --hparams "{\"lr\":1e-4, \"batch_size\":512, \"input_shape\":224, \"T\":9, \"tc_temperature\":9, \"index_dataset\":1, \"model\": \"${STS}\", \"resnet50_augmix\":0, \"vit\":1, \"clip\":0, \"dinov2\":1, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --dinov\
       --seed $INPUT_SEED\
       --trial_seed $INPUT_SEED\
       --subset_holdout_fraction ${INPUT_SUBSET_ST}\
       --start_step 10001\
       --steps 20001\
       --checkpoint_freq 1000\
       --test_envs 2\
       --output_dir /scratch/jl4476/${DATASET}/st_dinov2/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
       --checkpoint /scratch/jl4476/${DATASET}/st_dinov2/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
       --checkpoint_algorithm StudentSingle\
       --freeze\
       --mse\
       --deterministic_only\

mkdir -p ${DATASET}/st_dinov2/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}/
cp /scratch/jl4476/${DATASET}/st_dinov2/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}/results.jsonl ${DATASET}/st_dinov2/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}/
       

python3 -m domainbed.scripts.eval_mse\
       --data_dir=/scratch/$USER/data/\
       --dataset ${DATASET}\
       --test_env 2\
       --output_dir /scratch/jl4476/${DATASET}/st_dinov2/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
       --checkpoint /scratch/jl4476/${DATASET}/st_dinov2/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
       --hparams "{\"lr\":1e-4, \"batch_size\":512, \"input_shape\":224, \"T\":9, \"tc_temperature\":9, \"index_dataset\":1, \"model\": \"${INPUT_TC}\", \"resnet\":0, \"resnet50_augmix\":0, \"vit\":1, \"clip\":0, \"dinov2\":1, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
