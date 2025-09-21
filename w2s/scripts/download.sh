#!/bin/bash
#SBATCH --job-name=download
#SBATCH --array=1
#SBATCH --output=%x_%A_%a.out           # output file name
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --time 2:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --error=%x_%A_%a.out            # error file name (same to watch just one file)

module purge
source $HOME/.bashrc
export PATH=/scratch/yl9315/miniconda3/bin:$PATH
module load cuda/11.6.2
conda activate DomainBed2

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

python3 -m domainbed.scripts.download \
       --data_dir=/vast/yl9315/data/Places365