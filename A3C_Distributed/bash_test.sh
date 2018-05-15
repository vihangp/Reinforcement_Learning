#!/bin/bash -l
#SBATCH --job-name=cluster_test
#SBATCH --time=23:50:59
#SBATCH --nodes=2
#SBATCH --output=cluster_test_out.%j.out
#SBATCH --exclusive
#SBATCH --error=cluster_test_error.%j.err
#SBATCH --partition=slim
# load modules

module load openmpi

mpirun -N 1 bash_test2.sh

