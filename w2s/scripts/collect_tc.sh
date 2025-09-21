#!/bin/bash
#SBATCH --job-name=collect_tc
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

python3 -m domainbed.scripts.collect_intrinsic_dim --output_dir tc_ColoredMNISTID

#python3 -m domainbed.scripts.collect_tc --output_dir tc_2

#python3 -m domainbed.scripts.tc_checkpoint --output_dir tc/0 --seed 0\

#python3 -m domainbed.scripts.tc_checkpoint --output_dir tc/1 --seed 1\

#python3 -m domainbed.scripts.tc_checkpoint --output_dir tc/2 --seed 2\

#python3 -m domainbed.scripts.tc_checkpoint --output_dir tc/3 --seed 3\

#python3 -m domainbed.scripts.tc_checkpoint --output_dir tc/4 --seed 4\
