#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./573_gpu

# install pytorch with CUDA enabled
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch --force-reinstall
