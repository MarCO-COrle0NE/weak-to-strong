#!/bin/bash
#SBATCH --job-name=st_nlp
#SBATCH --array=106-140
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 24:00:00              # maximum execution time (HH:MM:SS)
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
#export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Define your input file lists
#TC=("prajjwal1/bert-tiny" "prajjwal1/bert-mini" "prajjwal1/bert-small" "prajjwal1/bert-medium")
TC=("prajjwal1/bert-tiny" "prajjwal1/bert-mini" "prajjwal1/bert-medium" )  # 3 teachers
SUBSET=(4000 8000 12000 16000)
SUBSET_ST=(20000 28000 36000 44000)

SUBSET=(500 1000 2000 3000)
SUBSET_ST=(3000 5000 10000 50000)

SUBSET=(500 1000 2000 3000 4000 8000 12000)
SUBSET_ST=(16000 24000 32000 40000 48000)
DATASET=("SST2") # 3 datasets

# Calculate the total number of combinations
# TOTAL_COMBINATIONS=$(24)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
# IDX_TC=$((TASK_ID / ${#SUBSET[@]})) # Get index for ST
# IDX_SUBSET=$((TASK_ID % ${#SUBSET[@]}))    # Get index for DATASET
IDX_TC=$((TASK_ID / $((${#SUBSET[@]} * ${#SUBSET_ST[@]})))) # Get index for ST
IDX_SUBSET=$(($((TASK_ID % $((${#SUBSET[@]} * ${#SUBSET_ST[@]})))) / ${#SUBSET_ST[@]})) # Get index for TC
IDX_SUBSET_ST=$((TASK_ID % ${#SUBSET_ST[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_DATASET="SST2"
INPUT_SUBSET=${SUBSET[$IDX_SUBSET]}
INPUT_SUBSET_INT=${SUBSET[$IDX_SUBSET]}
INPUT_SUBSET_ST=$INPUT_N
INPUT_TC=${TC[$IDX_TC]}
INPUT_ST="bert-base-uncased"
#INPUT_ST="roberta-base"
SEED=0
INPUT_N=${SUBSET_ST[$IDX_SUBSET_ST]}

mkdir -p /scratch/jl4476/data/SST2/
rsync -av --update ./domainbed/data/SST-2/ /scratch/$USER/data/SST2/

python3 -m domainbed.scripts.st_nlp\
       --data_dir=/scratch/jl4476/data/SST2/\
       --data_name ${INPUT_TC}_${INPUT_SUBSET_INT}.tsv\
       --dataset ${INPUT_DATASET}\
       --model_name $INPUT_ST\
       --hparams "{\"lr\":3e-5, \"batch_size\":16, \"weight_decay\":0.00001, \"warmup_steps\":0}"\
       --n $INPUT_SUBSET_INT\
       --N $INPUT_N\
       --test_length 7349\
       --epoch 50\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/st/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_N}_${INPUT_ST}\

#rsync -av --ignore-existing /scratch/$USER/data/SST2/ ./domainbed/data/SST-2/ 

# accelerate launch --multi_gpu -m domainbed.scripts.st_nlp_acc\
#        --data_dir=/scratch/jl4476/data/SST2/\
#        --data_name ${INPUT_TC}_${INPUT_SUBSET_INT}.tsv\
#        --dataset ${INPUT_DATASET}\
#        --model_name $INPUT_TC\
#        --hparams "{\"lr\":3e-5, \"batch_size\":16, \"weight_decay\":0.0001, \"warmup_steps\":100}"\
#        --n $INPUT_SUBSET_INT\
#        --N $INPUT_N\
#        --test_length 7349\
#        --epoch 4\
#        --test_envs 2\
#        --seed $SEED\
#        --trial_seed $SEED\
#        --output_dir /scratch/jl4476/${INPUT_DATASET}/st/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_N}_${INPUT_ST}\

mkdir -p /auto/u/jl4476/w2s/${INPUT_DATASET}/st/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST}_${INPUT_ST}/
cp -rf /scratch/jl4476/${INPUT_DATASET}/st/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST}_${INPUT_ST}/logs /auto/u/jl4476/w2s/${INPUT_DATASET}/st/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST}_${INPUT_ST}/