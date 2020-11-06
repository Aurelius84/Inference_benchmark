#!/bin/bash

original_model_root_dir=$1
mount_dir_name="models"

repo_dir=$(cd "$(dirname "$0")";cd ../..;pwd)
mount_repo_name="inference_benchmark"


# pull official images
sudo docker pull floopcz/tensorflow_cc:ubuntu-cuda

# create container
sudo nvidia-docker run --name tf_dy2stat_infer_cuda10 --net=host -v ${mount_model_root_dir}:/workspace/$mount_dir_name -v ${repo_dir}:/workspace/$mount_repo_name -it floopcz/tensorflow_cc:ubuntu-cuda  /bin/bash