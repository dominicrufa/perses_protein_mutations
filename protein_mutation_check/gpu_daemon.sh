#!/bin/bash

#Set the job name and wall time limit
#BSUB -J "daemon"
#BSUB -W 4:00


# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e

# Set gpu options.
#BSUB -n 1 -R "rusage[mem=8]"
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -L /bin/bash
export PATH="/home/rufad/miniconda3/envs/openmm/bin:$PATH"
#quit on first error
set -e 

cd $LS_SUBCWD
module load cuda/9.2
#python run_pickled_htf.py ALA_SER.vacuum.default_map.pkl &> ALA_SER.vacuum.default_map.log
#python run_pickled_htf.py SER_CYS.vacuum.default_map.pkl &> SER_CYS.vacuum.default_map.log
#python run_pickled_htf.py CYS_ALA.vacuum.default_map.pkl &> CYS_ALA.vacuum.default_map.log

#python run_pickled_htf.py SER_ALA.vacuum.default_map.pkl &> SER_ALA.vacuum.default_map.log
#python run_pickled_htf.py CYS_SER.vacuum.default_map.pkl &> CYS_SER.vacuum.default_map.log
python run_pickled_htf.py ALA_CYS.vacuum.default_map.pkl &> ALA_CYS.vacuum.default_map.log
