#!/bin/bash
#SBATCH --job-name=tc_intrinsic_did_vit
#SBATCH --array=2,4,6,8
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 48:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%A_%a.out            # error file name (same to watch just one file)

module purge
source $HOME/.bashrc
export PATH=/scratch/yl9315/miniconda3/bin:$PATH
#module load cuda/11.6.2
conda activate DomainBed

# Deterministic
#export CUBLAS_WORKSPACE_CONFIG=:4096:8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Define your input file lists
TC=("vit" "vit")  # 3 teachers
ARCH=("tiny" "large")
#SUBSET=(1 0.02 0.05 0.07 0.1 0.2 0.5 0.75 1)
SUBSET=(1)
#INTRINSIC_DIM=(100 200 300 400)
INTRINSIC_DIM=(60 80 100 120)
#SUBSET=(0.5 0.25 0.125)  # 3 students
DATASET=("ColoredMNISTID") 
LR=(1e-2 1e-2)
BATCH=(512 64)
MOMENTUM=(0.9 0.9)

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(16)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
IDX_DIM=$((TASK_ID / ${#TC[@]})) # Get index for ST
IDX_TC=$((TASK_ID % ${#TC[@]}))    # Get index for DATASET

# Get the corresponding input files
INPUT_DATASET=${DATASET[0]}
INPUT_DIM=${INTRINSIC_DIM[$IDX_DIM]}
#INPUT_SUBSET=${SUBSET[0]}
INPUT_SUBSET=2
INPUT_TC=${TC[$IDX_TC]}
INPUT_LR=${LR[$IDX_TC]}
INPUT_MOMENTUM=${MOMENTUM[$IDX_TC]}
INPUT_ARCH=${ARCH[$IDX_TC]}
INPUT_BATCH=${BATCH[$IDX_TC]}


python3 -m domainbed.scripts.tc\
       --data_dir=/vast/yl9315/data/Places365\
       --dataset ${INPUT_DATASET}\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":$INPUT_LR, \"SGD\":1, \"momentum\":$INPUT_MOMENTUM, \"batch_size\":$INPUT_BATCH, \"input_shape\":224, \"resnet50_augmix\":0, \"model\": \"${INPUT_TC}\", \"${INPUT_TC}\":1, \"arch\": \"${INPUT_ARCH}\", \"dino\":0, \"dinov2\":0, \"clip\":0}"\
       --checkpoint_freq 500\
       --steps 10001\
       --start_step 1\
       --test_envs 2\
       --intrinsic_dimension $INPUT_DIM\
       --output_dir ${INPUT_DATASET}/tc_intrinsic_did/${INPUT_SUBSET}_${INPUT_TC}_${INPUT_ARCH}_${INPUT_DIM}\
       
python3 -m domainbed.scripts.tc\
       --data_dir=/vast/yl9315/data/Places365\
       --dataset ${INPUT_DATASET}\
       --subset_holdout_fraction ${INPUT_SUBSET}\
       --hparams "{\"lr\":$INPUT_LR, \"SGD\":1, \"momentum\":$INPUT_MOMENTUM, \"batch_size\":$INPUT_BATCH, \"input_shape\":224, \"resnet50_augmix\":0, \"model\": \"${INPUT_TC}\", \"${INPUT_TC}\":1, \"arch\": \"${INPUT_ARCH}\", \"dino\":0, \"dinov2\":0, \"clip\":0}"\
       --checkpoint_freq 1000\
       --steps 20001\
       --start_step 10001\
       --test_envs 2\
       --intrinsic_dimension $INPUT_DIM\
       --output_dir ${INPUT_DATASET}/tc_intrinsic_did/${INPUT_SUBSET}_${INPUT_TC}_${INPUT_ARCH}_${INPUT_DIM}\
       --checkpoint ${INPUT_DATASET}/tc_intrinsic_did/${INPUT_SUBSET}_${INPUT_TC}_${INPUT_ARCH}_${INPUT_DIM}\
       --checkpoint_erm
       
       
       #--indices_dir ${INPUT_DATASET}/indices\