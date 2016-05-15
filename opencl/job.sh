#!/bin/bash
#SBATCH -J Test        # Job Name
#SBATCH -o Test.o%j    # Output and error file name (%j expands to jobID)
#SBATCH -n 16           # Total number of  tasks requested
#SBATCH -p gpudev  # Queue (partition) name -- normal, development, etc.
#SBATCH -t 01:00:00     # Run time (hh:mm:ss) - 1.5 hours

N_THREADS=1024 N_VERTICES=8192

for k in 2 4
do
    ./kmeans -o -i inp_100000_2.txt -n $k -m $k >> out
done
