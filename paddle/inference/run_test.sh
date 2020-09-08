#!/bin/bash
WITH_MKL=ON
WITH_GPU=ON
USET_TENSORRT=OFF

CUDA_LIB_DIR='/usr/local/cuda-9.0/lib64/'
CUDNN_LIB_DIR='/usr/lib/x86_64-linux-gnu/'
LIB_DIR='/workspace/code_dev/paddle-fork/build_infer/fluid_inference_install_dir'

DEMO_NAME='image_classification'
MODEL_DIR='/workspace/code_dev/paddle-predict/mobilenetv1'


mkdir -p build
cd build
rm -rf *

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDA_LIB=${CUDA_LIB_DIR} \
  -DCUDNN_LIB=${CUDNN_LIB_DIR}

make -j4
cp image_classification ../