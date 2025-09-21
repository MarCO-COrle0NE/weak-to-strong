#!/bin/bash
#SBATCH --job-name=tc_dsw_resnet18
#SBATCH --array=1-18
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

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Deterministic
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Define your input file lists
TC=("resnet18")  # 3 teachers
#SUBSET=(0.00273 0.0036 0.005 0.01 0.015 0.02 0.02322)
#SUBSET_INT=(75 100 140 280 420 560 650)
# D_S=25
# D_W=133
# DSW=(0 5 10 15 20 25)
# D_S=96
# D_W=234
# DSW=(0 20 40 60 80 96)
D_S=218
D_W=317
DSW=(0 40 80 120 160 218)
# D_S=218
# D_W=32
# DSW=(0 6 12 18 24 32)
# SUBSET=(0.00714 0.02143 0.03571 0.05 0.0643)
# SUBSET_INT=(200 600 1000 1400 1800)
# SUBSET=(0.01)
# SUBSET_INT=(280)
SUBSET=(0.00273 0.01 0.02)
SUBSET_INT=(75 280 560)
SUBSET_INT=(350 420 490)
#SUBSET_ST=(62925 62720 62440)
SUBSET_ST=(62650 62580 62510)
#SUBSET=(0.5 0.25 0.125)  # 3 students
DATASET=("ColoredMNISTID2") # 3 datasets

# Calculate the total number of combinations
# TOTAL_COMBINATIONS=$(24)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_SUBSET=$((TASK_ID / ${#DSW[@]})) # Get index for ST
IDX_DSW=$((TASK_ID % ${#DSW[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_DATASET=${DATASET[0]}
INPUT_SUBSET=${SUBSET[$IDX_SUBSET]}
INPUT_SUBSET_INT=${SUBSET_INT[$IDX_SUBSET]}
INPUT_SUBSET_ST=${SUBSET_ST[$IDX_SUBSET]}
INPUT_TC=${TC[0]}
INPUT_DSW=${DSW[$IDX_DSW]}
SEED=0
W_DIR="${INPUT_DATASET}/W/${INPUT_TC}"

#mkdir -p /scratch/jl4476/${W_DIR}
mkdir -p /scratch/jl4476/${INPUT_DATASET}/indices
mkdir -p /scratch/jl4476/data
rsync -av --ignore-existing ./domainbed/data/MNIST/ /scratch/$USER/data/
rsync -av --ignore-existing ./${INPUT_DATASET}/indices/ /scratch/$USER/${INPUT_DATASET}/indices/
#rsync -av --ignore-existing ./${W_DIR}/ /scratch/jl4476/${W_DIR}/

python3 -m domainbed.scripts.tc2\
       --data_dir=/scratch/jl4476/data/\
       --dataset ${INPUT_DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":1e-3, \"batch_size\":512, \"seed\":$SEED,\"st_length\":$INPUT_SUBSET_ST,\"tc_length\":$INPUT_SUBSET_INT,\"w_or_s\":\"w\", \"d\":2048,\"d_s\":$D_S, \"d_w\":$D_W, \"d_sw\":$INPUT_DSW, \"W_dir\":\"/scratch/jl4476/${W_DIR}\", \"input_shape\":224, \"resnet18\":1, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"18\"}"\
       --checkpoint_freq 1000\
       --steps 10001\
       --start_step 1\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --freeze\
       --mse\
       --deterministic_only\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/tc_dw/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${D_W}_${INPUT_DSW}\

python3 -m domainbed.scripts.tc2\
       --data_dir=/scratch/jl4476/data/\
       --dataset ${INPUT_DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":5e-4,  \"batch_size\":512, \"seed\":$SEED,\"st_length\":$INPUT_SUBSET_ST,\"tc_length\":$INPUT_SUBSET_INT,\"w_or_s\":\"w\",\"d\":2048, \"d_s\":$D_S, \"d_w\":$D_W, \"d_sw\":$INPUT_DSW, \"W_dir\":\"/scratch/jl4476/${W_DIR}\", \"input_shape\":224, \"resnet18\":1, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"18\"}"\
       --checkpoint_freq 1000\
       --steps 20001\
       --start_step 10001\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --freeze\
       --mse\
       --deterministic_only\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/tc_dw/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${D_W}_${INPUT_DSW}\
       --checkpoint /scratch/jl4476/${INPUT_DATASET}/tc_dw/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${D_W}_${INPUT_DSW}\
       --checkpoint_erm\

python3 -m domainbed.scripts.tc2\
       --data_dir=/scratch/jl4476/data/\
       --dataset ${INPUT_DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":1e-4,  \"batch_size\":512, \"seed\":$SEED,\"st_length\":$INPUT_SUBSET_ST,\"tc_length\":$INPUT_SUBSET_INT,\"w_or_s\":\"w\",\"d\":2048, \"d_s\":$D_S, \"d_w\":$D_W, \"d_sw\":$INPUT_DSW, \"W_dir\":\"/scratch/jl4476/${W_DIR}\", \"input_shape\":224, \"resnet18\":1, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"18\"}"\
       --checkpoint_freq 1000\
       --steps 30001\
       --start_step 20001\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --freeze\
       --mse\
       --deterministic_only\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/tc_dw/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${D_W}_${INPUT_DSW}\
       --checkpoint /scratch/jl4476/${INPUT_DATASET}/tc_dw/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${D_W}_${INPUT_DSW}\
       --checkpoint_erm\

python3 -m domainbed.scripts.tc2\
       --data_dir=/scratch/jl4476/data/\
       --dataset ${INPUT_DATASET}\
       --indices_dir /scratch/jl4476/${INPUT_DATASET}/indices\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":3e-5, \"batch_size\":512, \"seed\":$SEED,\"st_length\":$INPUT_SUBSET_ST,\"tc_length\":$INPUT_SUBSET_INT,\"w_or_s\":\"w\",\"d\":2048, \"d_s\":$D_S, \"d_w\":$D_W, \"d_sw\":$INPUT_DSW, \"W_dir\":\"/scratch/jl4476/${W_DIR}\", \"input_shape\":224, \"resnet18\":1, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"18\"}"\
       --checkpoint_freq 1000\
       --steps 50001\
       --start_step 30001\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --freeze\
       --mse\
       --deterministic_only\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/tc_dw/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${D_W}_${INPUT_DSW}\
       --checkpoint /scratch/jl4476/${INPUT_DATASET}/tc_dw/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${D_W}_${INPUT_DSW}\
       --checkpoint_erm\

python3 -m domainbed.scripts.diction_single\
       --data_dir=/scratch/jl4476/data/\
       --teacher_dir /scratch/jl4476/${INPUT_DATASET}/tc_dw/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${D_W}_${INPUT_DSW}\
       --hparams "{\"lr\":3e-3, \"batch_size\":512,\"seed\":$SEED,\"st_length\":$INPUT_SUBSET_ST,\"tc_length\":$INPUT_SUBSET_INT, \"w_or_s\":\"w\",\"d\":2048, \"d_s\":$D_S, \"d_w\":$D_W, \"d_sw\":$INPUT_DSW, \"W_dir\":\"/scratch/jl4476/${W_DIR}\", \"input_shape\":224, \"resnet18\":1, \"resnet50_augmix\":0, \"vit\":0, \"model\": \"${INPUT_TC}\", \"mobile\":0, \"alexnet\":0, \"mamba\":0, \"arch\": \"18\"}"\
       --dataset ${INPUT_DATASET}\
       --test_env 1\
       --seed $SEED\
       --trial_seed $SEED\
       --deterministic_only\
       --teacher_name model.pkl\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/diction/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${D_W}_${INPUT_DSW}\
       
mkdir -p ${INPUT_DATASET}/diction/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${D_W}_${INPUT_DSW}_atmpt3/
cp /scratch/$USER/${INPUT_DATASET}/diction/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${D_W}_${INPUT_DSW}/diction.json ./${INPUT_DATASET}/diction/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${D_W}_${INPUT_DSW}_atmpt3/
       