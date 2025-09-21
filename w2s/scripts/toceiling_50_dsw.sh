#!/bin/bash
#SBATCH --job-name=toceiling_ds_resnet50
#SBATCH --array=1-5
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 8:00:00              # maximum execution time (HH:MM:SS)
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
ST=("50")
DATASETS=("ColoredMNISTID") 
LRS=(3e-3)
SUBSET=(0.325)
SUBSET_INT=(9100)
SEED=(0)
D_S=(25 36 55 96 218)

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(160)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_D_S=$((TASK_ID / $((${#ST[@]} * ${#SEED[@]})))) # Get index for ST
IDX_ST=$(($((TASK_ID % $((${#ST[@]} * ${#SEED[@]})))) / ${#SEED[@]})) # Get index for TC
IDX_SEED=$((TASK_ID % ${#SEED[@]}))    # Get index for DATASET
IDX_DATASET=0

# Get the corresponding input files
INPUT_SEED=${SEED[$IDX_SEED]}
INPUT_ST=${ST[$IDX_ST]}
INPUT_SUBSET=${SUBSET[0]}
INPUT_SUBSET_INT=${SUBSET_INT[0]}
INPUT_TC=${TC[0]}
INPUT_D_S=${D_S[$IDX_D_S]}
DATASET=${DATASETS[$IDX_DATASET]}
LR=${LRS[$IDX_DATASET]}
       
mkdir -p /scratch/jl4476/${DATASET}/indices
mkdir -p /scratch/jl4476/data
rsync -av --ignore-existing ./domainbed/data/MNIST/ /scratch/$USER/data/
rsync -av --ignore-existing ./${DATASET}/indices/ /scratch/$USER/${DATASET}/indices/


python3 -m domainbed.scripts.ceiling_single\
       --data_dir=/scratch/$USER/data/\
       --dataset ${DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --hparams "{\"lr\":1e-3, \"batch_size\":512, \"input_shape\":224, \"w_or_s\":\"s\", \"d_s\":$INPUT_D_S, \"d_w\":133, \"d_sw\":$INPUT_D_S, \"model\": \"${STS}\", \"resnet18\":0, \"vit\":0, \"resnet50_augmix\":1, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --seed $INPUT_SEED\
       --trial_seed $INPUT_SEED\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --start_step 1\
       --steps 5001\
       --checkpoint_freq 1000\
       --test_envs 2\
       --output_dir /scratch/jl4476/${DATASET}/toceiling_dsw/${INPUT_SEED}/${INPUT_SUBSET_INT}_${STS}_${INPUT_ST}_${INPUT_D_S}\
       --freeze\
       --mse\

python3 -m domainbed.scripts.ceiling_single\
       --data_dir=/scratch/$USER/data/\
       --dataset ${DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --hparams "{\"lr\":5e-4, \"batch_size\":512, \"input_shape\":224, \"w_or_s\":\"s\", \"d_s\":$INPUT_D_S, \"d_w\":133, \"d_sw\":$INPUT_D_S, \"model\": \"${STS}\", \"resnet18\":0, \"vit\":0, \"resnet50_augmix\":1, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --seed $INPUT_SEED\
       --trial_seed $INPUT_SEED\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --start_step 5001\
       --steps 10001\
       --checkpoint_freq 1000\
       --test_envs 2\
       --output_dir /scratch/jl4476/${DATASET}/toceiling_dsw/${INPUT_SEED}/${INPUT_SUBSET_INT}_${STS}_${INPUT_ST}_${INPUT_D_S}\
       --checkpoint /scratch/jl4476/${DATASET}/toceiling_dsw/${INPUT_SEED}/${INPUT_SUBSET_INT}_${STS}_${INPUT_ST}_${INPUT_D_S}\
       --load_head\
       --freeze\
       --mse\

python3 -m domainbed.scripts.ceiling_single\
       --data_dir=/scratch/$USER/data/\
       --dataset ${DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --hparams "{\"lr\":1e-4, \"batch_size\":512, \"input_shape\":224, \"w_or_s\":\"s\", \"d_s\":$INPUT_D_S, \"d_w\":133, \"d_sw\":$INPUT_D_S, \"model\": \"${STS}\", \"resnet18\":0, \"vit\":0, \"resnet50_augmix\":1, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --seed $INPUT_SEED\
       --trial_seed $INPUT_SEED\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --start_step 10001\
       --steps 20001\
       --checkpoint_freq 1000\
       --test_envs 2\
       --output_dir /scratch/jl4476/${DATASET}/toceiling_dsw/${INPUT_SEED}/${INPUT_SUBSET_INT}_${STS}_${INPUT_ST}_${INPUT_D_S}\
       --checkpoint /scratch/jl4476/${DATASET}/toceiling_dsw/${INPUT_SEED}/${INPUT_SUBSET_INT}_${STS}_${INPUT_ST}_${INPUT_D_S}\
       --load_head\
       --freeze\
       --mse\

python3 -m domainbed.scripts.ceiling_single\
       --data_dir=/scratch/$USER/data/\
       --dataset ${DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --hparams "{\"lr\":1e-5, \"batch_size\":512, \"input_shape\":224, \"w_or_s\":\"s\", \"d_s\":$INPUT_D_S, \"d_w\":133, \"d_sw\":$INPUT_D_S, \"model\": \"${STS}\", \"resnet18\":0, \"vit\":0, \"resnet50_augmix\":1, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --seed $INPUT_SEED\
       --trial_seed $INPUT_SEED\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --start_step 20001\
       --steps 30001\
       --checkpoint_freq 1000\
       --test_envs 2\
       --output_dir /scratch/jl4476/${DATASET}/toceiling_dsw/${INPUT_SEED}/${INPUT_SUBSET_INT}_${STS}_${INPUT_ST}_${INPUT_D_S}\
       --checkpoint /scratch/jl4476/${DATASET}/toceiling_dsw/${INPUT_SEED}/${INPUT_SUBSET_INT}_${STS}_${INPUT_ST}_${INPUT_D_S}\
       --load_head\
       --freeze\
       --mse\

       
       
       #--deterministic_only
       #--checkpoint ${DATASET}/st_mamba/${INPUT_SEED}/${INPUT_SUBSET}_${INPUT_TC}_${INPUT_SUBSET_ST}_${INPUT_ST}\
       #--checkpoint_algorithm StudentSingle\