#!/bin/bash
#SBATCH --job-name=collect_indices
#SBATCH --output=%x_%j.out           # output file name
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --time 2:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%j.out            # error file name (same to watch just one file)

module load anaconda3/2024.02
export HF_HOME=/scratch/jl4476
source $HOME/.bashrc
#export PATH=/scratch/yl9315/miniconda3/bin:$PATH
conda activate Domain

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

python3 -m domainbed.scripts.collect_indices\
       --data_dir=/scratch/jl4476/data/\
       --dc_dir PACS/diction_50/diction.json\
       --algorithm ERM\
       --dataset CIFAR10\
       --task domain_adaptation\
       --hparams "{\"lr\":1e-3, \"T\":9, \"dc_temperature\":15, \"tc_temperature\":9, \"index_dataset\":1, \"resnet50_augmix\":1, \"collect_indices\":1}"\
       --freeze\
       --steps 1200\
       --train_only\
       --all_envs\
       --checkpoint_freq 25\
       --test_envs 1\
       --output_dir CIFAR10/indices\
       