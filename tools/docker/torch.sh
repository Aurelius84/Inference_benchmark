#!/bin/bash

original_model_root_dir=$1
mount_dir_name="models"

repo_dir=$(cd "$(dirname "$0")";cd ../..;pwd)
mount_repo_name="inference_benchmark"

# pull official images
sudo docker pull pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

# create container
sudo nvidia-docker run --name torch_dy2stat_infer_cuda10 --net=host -v ${original_model_root_dir}:/workspace/$mount_dir_name -v ${repo_dir}:/workspace/$mount_repo_name -it pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel  /bin/bash

# update and download libtorch
apt-get update & apt-get install wget git -y
cd /workspace
wget https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcu101.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.6.0+cu101.zip
