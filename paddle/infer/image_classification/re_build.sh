#!/bin/bash

WITH_MKL=ON
export WITH_GPU=ON
USE_TENSORRT=OFF

if [[ -z "${PADDLE_LIB_DIR}" ]]; then
  echo -e "\033[33m ====> {PADDLE_LIB_DIR} is undefined, set to default value \033[0m";
  LIB_DIR='/workspace/code_dev/paddle-fork/build_infer_10/paddle_inference_install_dir';
else
  LIB_DIR=${PADDLE_LIB_DIR};
fi

if [[ -z "${CUDA_LIB_DIR}" ]]; then
  echo -e "\033[33m ====> {CUDA_LIB_DIR} is undefined, set to default value \033[0m";
  CUDA_LIB_DIR='/usr/local/cuda-10.1/lib64/';
fi

if [[ -z "${CUDNN_LIB_DIR}" ]]; then
  echo -e "\033[33m ====> {CUDNN_LIB_DIR} is undefined, set to default value \033[0m";
  CUDNN_LIB_DIR='/usr/lib/x86_64-linux-gnu/';
fi

# please modify LIB_DIR for your built inference lib dir
# TODO: Need discuss with QA to determine how to config this env vaiable
echo -e "\033[33m ====> LIB_DIR : ${LIB_DIR} \033[0m"
echo -e "\033[33m ====> CUDA_LIB_DIR : ${CUDA_LIB_DIR} \033[0m"
echo -e "\033[33m ====> CUDNN_LIB_DIR : ${CUDNN_LIB_DIR} \033[0m"

mkdir -p build
cd build
rm -rf *

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDA_LIB=${CUDA_LIB_DIR} \
  -DCUDNN_LIB=${CUDNN_LIB_DIR}

make -j4

