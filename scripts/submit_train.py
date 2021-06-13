import os

batch = """#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1
##SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=siddharthmishra19@gmail.com

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-astrometry/
"""

##########################
# Explore configurations #
##########################

batch_size_list = [128]
activations = ["relu"]
kernel_size_list = [4]
n_neighbours_list = [8]
laplacian_types = ["combinatorial"]
conv_types = ["chebconv"]
conv_source_list = ["deepsphere", "geometric"]

for n_neighbours in n_neighbours_list:
    for batch_size in batch_size_list:
        for activation in activations:
            for kernel_size in kernel_size_list:
                for laplacian_type in laplacian_types:
                    for conv_type in conv_types:
                        for conv_source in conv_source_list:
                            batchn = batch + "\n"
                            batchn += "python -u train.py --sample train --name prototype --batch_size {} --activation {} --kernel_size {} --laplacian_type {} --conv_type {} --n_neighbours {} --conv_source {}".format(batch_size, activation, kernel_size, laplacian_type, conv_type, n_neighbours, conv_source)
                            fname = "batch/submit.batch"
                            f = open(fname, "w")
                            f.write(batchn)
                            f.close()
                            os.system("chmod +x " + fname)
                            os.system("sbatch " + fname)