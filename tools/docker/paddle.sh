#!/bin/bash

# should specific this for loading model of @to_static
mount_model_root_dir=$1

# pull official image
docker images pull hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7

# create container
nvidia-docker run --name paddle_dy2stat_infer_cuda10 --net=host -v ${mount_model_root_dir}:/workspace -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  /bin/bash

apt-get update & apt install libprotobuf-dev

# attach container and prepare code
cd /workspace

# clone paddle and build infer lib
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
mkdir build_infer and cd build_infer
touch re_build.sh

echo "#!/bin/bash" >> re_build.sh
echo "cmake .. -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=ON -DWITH_GPU=ON -DON_INFER=ON -DWITH_PATHON=ON -DWITH_MKLDNN=OFF" >> re_build.sh
echo "make -j$(nproc)" >> re_build.sh
echo "make inference_lib_dist -j$(nproc)" >> re_build.sh

# clone benchmark code
git clone https://github.com/Aurelius84/Inference_benchmark.git
