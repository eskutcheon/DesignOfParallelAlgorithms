#!/bin/bash
#SBATCH --account=193000-cf0003
#SBATCH --job-name=fluid256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --no-reque
#SBATCH --partition=service
#SBATCH --qos=debug
module load cuda/11.2.1
./fluidcu -n 256 -stopTime 5 -o ftke256.dat >& fluidcu256.out

