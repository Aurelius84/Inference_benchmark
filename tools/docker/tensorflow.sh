#!/bin/bash

# should specific this for loading model of @to_static
mount_model_root_dir=$1

# pull official images
docker pull floopcz/tensorflow_cc:ubuntu-cuda

# create container
nvidia-docker run --name tf_dy2stat_infer_cuda10 --net=host -v ${mount_model_root_dir}:/workspace -it floopcz/tensorflow_cc:ubuntu-cuda  /bin/bash

# download repo
git clone https://github.com/Aurelius84/Inference_benchmark.git