#!/bin/bash -v

paddle_dir=/workspace/Paddle
build_dir=$paddle_dir/build_infer

if [ ! -d "$paddle_dir" ];
then
    apt-get update & apt install libprotobuf-dev
    wait

    # set env variables for build
    echo "export WITH_MKL=ON" >> ~/.bash_profile
    echo "export WITH_GPU=ON" >> ~/.bash_profile
    echo "USET_TENSORRT=OFF" >> ~/.bash_profile

    echo "CUDA_LIB_DIR='/usr/local/cuda-10.1/lib64/'" >> ~/.bash_profile
    echo "CUDNN_LIB_DIR='/usr/lib/x86_64-linux-gnu/'" >> ~/.bash_profile
    # please modify LIB_DIR for your built inference lib dir
    echo "LIB_DIR='$build_dir/fluid_inference_install_dir'" >> ~/.bash_profile

    # attach container and prepare code
    cd /workspace

    # clone paddle and build infer lib
    git clone https://github.com/PaddlePaddle/Paddle.git
    wait

    cd Paddle
    mkdir build_infer and cd build_infer

fi

if [ ! -f "$build_dir/re_build.sh" ]
then

    cd $build_dir
    touch re_build.sh

    echo "#!/bin/bash" >> re_build.sh
    echo "cmake .. -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=ON -DWITH_GPU=ON -DON_INFER=ON -DWITH_PATHON=ON -DWITH_MKLDNN=OFF" >> re_build.sh
    echo "make -j$(nproc)" >> re_build.sh
    echo "make inference_lib_dist -j$(nproc)" >> re_build.sh
fi

# apply latest code
cd $build_dir
git pull

# re build paddle lib
bash re_build.sh
