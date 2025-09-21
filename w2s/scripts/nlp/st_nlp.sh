#!/bin/bash
#SBATCH --job-name=st_nlp
#SBATCH --array=1-48
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gpus-per-node=4
#SBATCH --time 8:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%A_%a.out            # error file name (same to watch just one file)
#SBATCH --export=SEED

module load anaconda3/2024.02
export HF_HOME=/scratch/jl4476
export TORCH_HOME=/scratch/jl4476
source $HOME/.bashrc
#export PATH=/scratch/yl9315/miniconda3/bin:$PATH
conda activate Domain

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
# MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
#MASTER_PORT=$(expr 10000 + $(echo -n $((SLURM_JOBID + $(SLURM_ARRAY_TASK_ID))) | tail -c 4))
#MASTER_PORT=$((10000 + $(printf "%04d" $((SLURM_JOBID + SLURM_ARRAY_TASK_ID)) | tail -c 4)))
# Generate a unique port per SLURM task using job ID and task ID
MASTER_PORT=$((10000 + ($SLURM_JOBID % 10000) + ($SLURM_ARRAY_TASK_ID % 1000)))

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

export NCCL_DEBUG=INFO               # Enable NCCL debugging (optional)
export PYTHONFAULTHANDLER=1          # Enable Python traceback on fault

export TOKENIZERS_PARALLELISM=false
# Deterministic
#export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Define your input file lists
TC=("gpt2" "gpt2-medium" "gpt2-large")
SUBSET=(500 1000 2000 3000)
SUBSET_ST=(3000 6000 9000 12000)
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
INPUT_N=${SUBSET_ST[$IDX_SUBSET_ST]}
INPUT_SUBSET_ST=$INPUT_N
INPUT_TC=${TC[$IDX_TC]}
INPUT_ST="gpt2-xl"
INPUT_WARMUP=$(($INPUT_N / 64))
#SEED=0

mkdir -p /scratch/jl4476/data/SST2/
# rm -rf /scratch/jl4476/data/SST2
# mkdir -p /scratch/jl4476/data/SST2/
(
  flock 9
  rm -rf /scratch/$USER/data/SST2/prajjwal1
  rm -rf /scratch/$USER/data/SST2/google
  rm -f /scratch/$USER/data/SST2/a*tsv
  rm -f /scratch/$USER/data/SST2/b*tsv
  rm -f /scratch/$USER/data/SST2/d*tsv
  rm -f /scratch/$USER/data/SST2/r*tsv
  rsync -av --update ./domainbed/data/SST-2/ /scratch/$USER/data/SST2/
) 9>/tmp/rsync.lock

# python3 -m domainbed.scripts.st_nlp_conf\
#        --data_dir=/scratch/jl4476/data/SST2/\
#        --data_name ${SEED}_${INPUT_TC}_${INPUT_SUBSET_INT}.tsv\
#        --dataset ${INPUT_DATASET}\
#        --model_name $INPUT_ST\
#        --hparams "{\"lr\":5e-5, \"batch_size\":32, \"weight_decay\":0.0, \"warmup_steps\":0}"\
#        --n $INPUT_SUBSET_INT\
#        --N $INPUT_N\
#        --test_length 4349\
#        --epoch 5\
#        --test_envs 2\
#        --seed $SEED\
#        --trial_seed $SEED\
#        --output_dir /scratch/jl4476/${INPUT_DATASET}/st/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_N}_${INPUT_ST}\

#rsync -av --ignore-existing /scratch/$USER/data/SST2/ ./domainbed/data/SST-2/ 

accelerate launch --multi_gpu --num_processes 4 --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT -m domainbed.scripts.st_nlp_acc\
       --data_dir=/scratch/jl4476/data/SST2/\
       --data_name ${SEED}_${INPUT_TC}_${INPUT_SUBSET_INT}.tsv\
       --dataset ${INPUT_DATASET}\
       --model_name $INPUT_ST\
       --hparams "{\"lr\":1e-5, \"batch_size\":2, \"weight_decay\":0.0, \"warmup_steps\":0, \"gradient_accumulation_steps\":4}"\
       --n $INPUT_SUBSET_INT\
       --N $INPUT_N\
       --test_length 4349\
       --epoch 2\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/st/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_N}_${INPUT_ST}\

# mkdir -p /auto/u/jl4476/w2s/${INPUT_DATASET}/st/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST}_${INPUT_ST}/
# cp -rf /scratch/jl4476/${INPUT_DATASET}/st/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST}_${INPUT_ST}/logs /auto/u/jl4476/w2s/${INPUT_DATASET}/st/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST}_${INPUT_ST}/