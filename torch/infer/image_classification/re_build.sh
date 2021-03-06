#!/bin/bash
rm -rf build
mkdir build
cd build
rm -f CMakeCache.txt

libtorch_path=/workspace/code_dev/paddle-predict/torch/libtorch/

cmake -DCMAKE_PREFIX_PATH=$libtorch_path ../
cmake --build . --config Release

cp *_exe ../