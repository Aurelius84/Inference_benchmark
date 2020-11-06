#!/bin/bash

# should specific this for loading model of @to_static
mount_model_root_dir=$1

# pull official images
docker pull pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

# create container
nvidia-docker run --name torch_dy2stat_infer_cuda10 --net=host -v ${mount_model_root_dir}:/workspace -it pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel  /bin/bash

# update and download libtorch
apt-get update & apt-get install wget git -y
cd /workspace
wget https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcu101.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.6.0+cu101.zip

# download repo
git clone https://github.com/Aurelius84/Inference_benchmark.git