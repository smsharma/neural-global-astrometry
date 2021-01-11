
import sys, os
import random
import numpy as np

batch='''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH --mem=8GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=sm8383@nyu.edu
#SBATCH -p hepheno

# MPI compilers
export MKL_LP64_ILP64=ilp64
source /opt/intel/compilers_and_libraries_2016.2.181/linux/bin/compilervars.sh intel64
source /opt/intel/impi/5.0.3.048/bin64/mpivars.sh

source ~/.bashrc
source /group/hepheno/smsharma/heptools/anaconda3/etc/profile.d/conda.sh
conda activate 

cd /group/hepheno/smsharma/Lensing-PowerSpectra/simulation/

'''

# For compact objects example

for imc in range(500):
    batchn = batch  + "\n"
    batchn += "python astrometry_compact_uniform_sim_interface.py --imc " + str(imc)
    fname = "batch/mc_" + str(imc) + ".batch" 
    f=open(fname, "w")
    f.write(batchn)
    f.close()
    os.system("chmod +x " + fname);
    os.system("sbatch " + fname);

# For CDM-like example

for imc in range(500):
    batchn = batch  + "\n"
    batchn += "python astrometry_sim_interface.py --imc " + str(imc)
    fname = "batch/mc_" + str(imc) + ".batch" 
    f=open(fname, "w")
    f.write(batchn)
    f.close()
    os.system("chmod +x " + fname);
    os.system("sbatch " + fname);
