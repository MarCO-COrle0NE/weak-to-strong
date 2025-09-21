#!/bin/bash
#SBATCH --job-name=tost_resnet_hard_50
#SBATCH --array=1-20
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --gpus-per-node=1
#SBATCH --time 5:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%A_%a.out            # error file name (same to watch just one file)

module load anaconda3/2024.02
export HF_HOME=/scratch/jl4476
export TORCH_HOME=/scratch/jl4476
source $HOME/.bashrc
#export PATH=/scratch/yl9315/miniconda3/bin:$PATH
conda activate Domain

# Deterministic
export CUBLAS_WORKSPACE_CONFIG=:4096:8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Define your input file lists
TC=("resnet18")  # 3 teachers
STS="resnet50"
ST=("50")
DATASETS=("ColoredMNISTID") 
LRS=(3e-3)
SUBSET_ST=(0.1 0.115 0.13 0.147 0.1625 0.195 0.2275 0.24375 0.26 0.2925 0.325)
SUBSET_ST_INT=(2800 3220 3640 4116 4550 5460 6370 6825 7280 8190 9100)

SUBSET_ST=(0.1625 0.195 0.2275 0.24375 0.26 0.2925 0.325)
SUBSET_ST_INT=(4550 5460 6370 6825 7280 8190 9100)

# SUBSET_ST=(0.4286 0.5357143 0.642857143 0.75 0.857143 1)
# SUBSET_ST_INT=(12000 15000 18000 21000 24000 28000)

SUBSET_ST=(0.375 0.482143 0.5893 0.69643)
SUBSET_ST_INT=(10500 13500 16500 19500)

SUBSET_ST=(0.375 0.482143 0.5893 0.69643)
SUBSET_ST_INT=(10500 13500 16500 19500)

# SUBSET_ST=(0.1 0.115 0.13 0.147)
# SUBSET_ST_INT=(2800 3220 3640 4116)
#SUBSET=(0.01)
#SUBSET_INT=(280)
#SUBSET_ST=(0.195)
#SUBSET_ST_INT=(5460)
SUBSET=(0.00273 0.005 0.01 0.015 0.02322)
SUBSET_INT=(75 140 280 420 650)
# SUBSET=(0.001)
# SUBSET_INT=(28)
SEED=(0)

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(160)

# Map SLURM_ARRAY_TASK_ID to a pair of indices (i, j)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-based index
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
       --hparams "{\"lr\":3e-3, \"batch_size\":512, \"input_shape\":224, \"T\":9, \"tc_temperature\":9, \"index_dataset\":1, \"model\": \"${STS}\", \"resnet50_augmix\":1, \"vit\":0, \"clip\":0, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --dinov\
       --seed $INPUT_SEED\
       --trial_seed $INPUT_SEED\
       --subset_holdout_fraction ${INPUT_SUBSET_ST}\
       --start_step 1\
       --steps 3001\
       --checkpoint_freq 100\
       --test_envs 2\
       --output_dir /scratch/jl4476/${DATASET}/st_resnet_hard/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
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
       --hparams "{\"lr\":3e-3, \"batch_size\":512, \"input_shape\":224, \"T\":9, \"tc_temperature\":9, \"index_dataset\":1, \"model\": \"${STS}\", \"resnet50_augmix\":1, \"vit\":0, \"clip\":0, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --dinov\
       --seed $INPUT_SEED\
       --trial_seed $INPUT_SEED\
       --subset_holdout_fraction ${INPUT_SUBSET_ST}\
       --start_step 3001\
       --steps 9001\
       --checkpoint_freq 1000\
       --test_envs 2\
       --output_dir /scratch/jl4476/${DATASET}/st_resnet_hard/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
       --checkpoint /scratch/jl4476/${DATASET}/st_resnet_hard/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
       --checkpoint_algorithm StudentSingle\
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
       --hparams "{\"lr\":1e-4, \"batch_size\":512, \"input_shape\":224, \"T\":9, \"tc_temperature\":9, \"index_dataset\":1, \"model\": \"${STS}\", \"resnet50_augmix\":1, \"vit\":0, \"clip\":0, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --dinov\
       --seed $INPUT_SEED\
       --trial_seed $INPUT_SEED\
       --subset_holdout_fraction ${INPUT_SUBSET_ST}\
       --start_step 9001\
       --steps 12001\
       --checkpoint_freq 1000\
       --test_envs 2\
       --output_dir /scratch/jl4476/${DATASET}/st_resnet_hard/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
       --checkpoint /scratch/jl4476/${DATASET}/st_resnet_hard/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
       --checkpoint_algorithm StudentSingle\
       --freeze\
       --mse\
       --deterministic_only\

mkdir -p ${DATASET}/st_resnet_hard/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}
cp /scratch/jl4476/${DATASET}/st_resnet_hard/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}/results.jsonl ./${DATASET}/st_resnet_hard/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}/

python3 -m domainbed.scripts.eval_mse\
       --data_dir=/scratch/$USER/data/\
       --dataset ${DATASET}\
       --test_env 1\
       --output_dir /scratch/jl4476/st_${DATASET}/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
       --checkpoint /scratch/jl4476/${DATASET}/st_resnet_hard/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
       --hparams "{\"lr\":1e-4, \"batch_size\":512, \"input_shape\":224, \"T\":9, \"tc_temperature\":9, \"index_dataset\":1, \"model\": \"${STS}\", \"resnet50_augmix\":1, \"vit\":0, \"clip\":0, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
       --checkpoint_erm\

# python3 -m domainbed.scripts.st\
#        --data_dir=/scratch/$USER/data/\
#        --dc_dir /scratch/jl4476/${DATASET}/diction/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}/diction.json\
#        --indices_dir /scratch/jl4476/${DATASET}/indices\
#        --algorithm StudentSingleHard\
#        --dataset ${DATASET}\
#        --task domain_adaptation\
#        --hparams "{\"lr\":3e-5, \"batch_size\":512, \"input_shape\":224, \"T\":9, \"tc_temperature\":9, \"index_dataset\":1, \"model\": \"${STS}\", \"resnet50_augmix\":1, \"vit\":0, \"clip\":0, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
#        --dinov\
#        --seed $INPUT_SEED\
#        --trial_seed $INPUT_SEED\
#        --subset_holdout_fraction ${INPUT_SUBSET_ST}\
#        --start_step 12001\
#        --steps 16001\
#        --checkpoint_freq 1000\
#        --test_envs 2\
#        --output_dir /scratch/jl4476/${DATASET}/st_resnet_hard/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
#        --checkpoint /scratch/jl4476/${DATASET}/st_resnet_hard/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
#        --checkpoint_algorithm StudentSingle\
#        --freeze\
#        --mse\
#        --deterministic_only\

# python3 -m domainbed.scripts.st\
#        --data_dir=/scratch/$USER/data/\
#        --dc_dir /scratch/jl4476/${DATASET}/diction/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}/diction.json\
#        --indices_dir /scratch/jl4476/${DATASET}/indices\
#        --algorithm StudentSingleHard\
#        --dataset ${DATASET}\
#        --task domain_adaptation\
#        --hparams "{\"lr\":3e-5, \"batch_size\":512, \"input_shape\":224, \"T\":9, \"tc_temperature\":9, \"index_dataset\":1, \"model\": \"${STS}\", \"resnet50_augmix\":1, \"vit\":0, \"clip\":0, \"dinov2\":0, \"dino\":0, \"arch\": \"${INPUT_ST}\"}"\
#        --dinov\
#        --seed $INPUT_SEED\
#        --trial_seed $INPUT_SEED\
#        --subset_holdout_fraction ${INPUT_SUBSET_ST}\
#        --start_step 16001\
#        --steps 20001\
#        --checkpoint_freq 1000\
#        --test_envs 2\
#        --output_dir /scratch/jl4476/${DATASET}/st_resnet_hard/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
#        --checkpoint /scratch/jl4476/${DATASET}/st_resnet_hard/${INPUT_SEED}/${INPUT_SUBSET_INT}_${INPUT_TC}_${INPUT_SUBSET_ST_INT}_${INPUT_ST}\
#        --checkpoint_algorithm StudentSingle\
#        --freeze\
#        --mse\
#        --deterministic_only\