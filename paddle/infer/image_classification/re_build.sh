#!/bin/bash

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
cp image_classification_exe ../
