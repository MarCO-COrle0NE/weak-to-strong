#!/bin/bash
#SBATCH --job-name=tc_var
#SBATCH --array=1-9
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gpus-per-node=1
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
MASTER_PORT=$((10000 + ($SLURM_JOBID % 10000) + ($SLURM_ARRAY_TASK_ID % 1000)))

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

export NCCL_DEBUG=INFO               # Enable NCCL debugging (optional)
export PYTHONFAULTHANDLER=1          # Enable Python traceback on fault

# Deterministic
#export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Define your input file lists
TC=("prajjwal1/bert-medium")
SUBSET=(9000 11000 13000)
#SUBSET_ST=(8000 14000 20000 25000 30000 40000)
FLIP_PROBS=(0.1 0.2 0.3)
DATASET=("SST2") # 3 datasets

# Calculate the total number of combinations
# TOTAL_COMBINATIONS=$(24)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_FLIP=$((TASK_ID / ${#SUBSET[@]})) # Get index for ST
IDX_SUBSET=$((TASK_ID % ${#SUBSET[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_DATASET="SST2"
INPUT_SUBSET=${SUBSET[$IDX_SUBSET]}
INPUT_SUBSET_INT=${SUBSET[$IDX_SUBSET]}
INPUT_FLIP=${FLIP_PROBS[$IDX_FLIP]}
INPUT_TC=${TC[0]}
#SEED=0
INPUT_N=$((60000 - $INPUT_SUBSET_INT))
INPUT_WARMUP=$(($INPUT_SUBSET_INT / 64))

mkdir -p /scratch/jl4476/data/SST2/
rsync -av --update ./domainbed/data/SST-2/ /scratch/$USER/data/SST2/

python3 -m domainbed.scripts.tc_nlp\
       --data_dir=/scratch/jl4476/data/SST2/\
       --dataset ${INPUT_DATASET}\
       --model_name $INPUT_TC\
       --hparams "{\"lr\":3e-5, \"batch_size\":32, \"weight_decay\":0.0001, \"warmup_steps\":$INPUT_WARMUP}"\
       --n $INPUT_SUBSET_INT\
       --N $INPUT_N\
       --test_length 4349\
       --epoch 30\
       --test_envs 2\
       --seed $SEED\
       --trial_seed $SEED\
       --save_labels\
       --flip_prob $INPUT_FLIP\
       --output_dir /scratch/jl4476/${INPUT_DATASET}/tc/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_FLIP}\

rsync -av --update /scratch/$USER/data/SST2/ ./domainbed/data/SST-2/ 
#cp /scratch/$USER/data/SST2/${INPUT_TC}_${INPUT_SUBSET_INT}.tsv ./domainbed/data/SST-2/

# accelerate launch --multi_gpu --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT -m domainbed.scripts.tc_nlp_acc\
#        --data_dir=/scratch/jl4476/data/SST2/\
#        --dataset ${INPUT_DATASET}\
#        --model_name $INPUT_TC\
#        --hparams "{\"lr\":3e-5, \"batch_size\":16, \"weight_decay\":0.00001, \"warmup_steps\":0}"\
#        --n $INPUT_SUBSET_INT\
#        --N $INPUT_N\
#        --test_length 7349\
#        --epoch 50\
#        --test_envs 2\
#        --seed $SEED\
#        --trial_seed $SEED\
#        --output_dir /scratch/jl4476/${INPUT_DATASET}/tc/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}\

mkdir -p /auto/u/jl4476/w2s/${INPUT_DATASET}/tc/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_2/
cp -rf /scratch/jl4476/${INPUT_DATASET}/tc/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}/logs /auto/u/jl4476/w2s/${INPUT_DATASET}/tc/${SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_2/