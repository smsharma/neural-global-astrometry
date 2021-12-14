import os
import numpy as np

batch = """#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=35:59:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=smsharma@mit.com

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/neural-global-astrometry/
"""

############################
# Grid over configurations #
############################

batch_size_list = [256]
activations = ["relu"]
kernel_size_list = [5, 8]
n_neighbours_list = [8]
laplacian_types = ["combinatorial"]
conv_types = ["chebconv"]
conv_source_list = ["deepsphere"]
sigma_noise_list = [0.01]  # [np.linspace(0.0002, 0.003, 5)[-2]]
pooling_end_list = ["average"]

for n_neighbours in n_neighbours_list:
    for batch_size in batch_size_list:
        for activation in activations:
            for kernel_size in kernel_size_list:
                for laplacian_type in laplacian_types:
                    for conv_type in conv_types:
                        for conv_source in conv_source_list:
                            for sigma_noise in sigma_noise_list:
                                for pooling_end in pooling_end_list:
                                    batchn = batch + "\n"
                                    batchn += "python -u train.py --sample train --name response_bugfix --batch_size {} --activation {} --kernel_size {} --laplacian_type {} --conv_type {} --n_neighbours {} --conv_source {} --sigma_noise {} --fc_dims '[[-1, 1024],[1024, 256]]' --numpy_noise 0 --pooling_end {} --sigma_noise_model_file gaia_DR2_quasar_noise.npy".format(batch_size, activation, kernel_size, laplacian_type, conv_type, n_neighbours, conv_source, sigma_noise, pooling_end)
                                    fname = "batch/submit.batch"
                                    f = open(fname, "w")
                                    f.write(batchn)
                                    f.close()
                                    os.system("chmod +x " + fname)
                                    os.system("sbatch " + fname)

#  --sigma_noise_model_file gaia_DR2_quasar_noise.npy