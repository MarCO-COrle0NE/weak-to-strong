#!/bin/bash
#SBATCH --job-name=collect_weak
#SBATCH --output=%x_%j.out           # output file name
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --time 2:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%j.out            # error file name (same to watch just one file)

module purge
source $HOME/.bashrc
export PATH=/scratch/yl9315/miniconda3/bin:$PATH
module load cuda/11.6.2
conda activate DomainBed

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

python3 -m domainbed.scripts.collect_weak_subset_seed --output_dir tc_ColoredMNISTID/0/resnet18 --seed 0\

#python3 -m domainbed.scripts.collect_weak --output_dir weak//3/s14 --seed 3\

#python3 -m domainbed.scripts.collect_weak --output_dir weak//4/s14 --seed 4\

#python3 -m domainbed.scripts.collect_weak --output_dir weak/2/s14 --seed 2\

#python3 -m domainbed.scripts.collect_weak --output_dir weak/1/s14 --seed 1\

#python3 -m domainbed.scripts.collect_weak --output_dir weak/0/s14 --seed 0\

#python3 -m domainbed.scripts.collect_weak --output_dir weak/3/50 --seed 3 --step 13500\

#python3 -m domainbed.scripts.collect_weak --output_dir weak/4/50 --seed 4 --step 13500\

#python3 -m domainbed.scripts.collect_weak --output_dir weak/2/50 --seed 2 --step 13500\

#python3 -m domainbed.scripts.collect_weak --output_dir weak/1/50 --seed 1 --step 13500\

#python3 -m domainbed.scripts.collect_weak --output_dir weak/0/50 --seed 0 --step 13500\

#python3 -m domainbed.scripts.collect_weak_18 --output_dir weak/3/18 --seed 3\

#python3 -m domainbed.scripts.collect_weak_18 --output_dir weak/4/18 --seed 4\

#python3 -m domainbed.scripts.collect_weak_18 --output_dir weak/2/18 --seed 2\

#python3 -m domainbed.scripts.collect_weak_18 --output_dir weak/1/18 --seed 1\

#python3 -m domainbed.scripts.collect_weak_18 --output_dir weak/0/18 --seed 0

#python3 -m domainbed.scripts.collect_weak --output_dir weak_subset/22 --job 51822252\

#python3 -m domainbed.scripts.collect_weak_steps --output_dir weak_steps/0/50 --seed 0\

#python3 -m domainbed.scripts.collect_weak_steps --output_dir weak_steps/1/50 --seed 1\

#python3 -m domainbed.scripts.collect_weak_steps --output_dir weak_steps/2/50 --seed 2\

#python3 -m domainbed.scripts.collect_weak_steps --output_dir weak_steps/3/50 --seed 3\

#python3 -m domainbed.scripts.collect_weak_steps --output_dir weak_steps/4/50 --seed 4\