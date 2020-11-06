#!/bin/bash -v

# should specific this for loading model of @to_static
original_model_root_dir=$1
mount_dir_name="models"

repo_dir=$(cd "$(dirname "$0")";cd ../..;pwd)
mount_repo_name="inference_benchmark"

# pull official image
sudo docker images pull hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7

# create container
sudo nvidia-docker run --name paddle_dy2stat_infer_cuda10 --net=host -v ${original_model_root_dir}:/workspace/$mount_dir_name -v ${repo_dir}:/workspace/$mount_repo_name -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  /bin/bash
