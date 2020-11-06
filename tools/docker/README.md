# Inference_benchmark


**测试环境：**

+ Paddle：官方cuda10镜像，2.0beta分支
+ Torch：官方cuda10镜像，1.6.0版本
+ TensorFlow：开源cuda10.1镜像，2.2.0版本
+ 机器：P40，C++ 每个batch下的预测耗时


### 一、环境准备  && 测试

#### 1. Paddle环境搭建

Paddle官方发布了cuda10的docker镜像，首先拉取images:
```bash
docker images pull hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7
```

根据images，构建容器，并进入此容器
```bash
sudo nvidia-docker run --name XXX --net=host -v $PWD:/workspace -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  /bin/bash
```

克隆paddle的仓库，编译预测库lib文件
```
git clone https://github.com/PaddlePaddle/Paddle.git

cd Paddle
mkdir build_infer

touch re_build.sh
```

在创建的`re_build.sh`里，添加如下内容：
```
#!/bin/bash

cmake .. -DWITH_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_MKL=ON \
        -DWITH_GPU=ON \
        -DON_INFER=ON \
        -DWITH_CONTRIB=OFF \
        -DWITH_XBYAK=OFF \
        -DWITH_PATHON=OFF \
        -DWITH_MKLDNN=OFF

make -j$(nproc)
make inference_lib_dist -j$(nproc)
```

执行脚本，编译预测库，最终会在`build_infer`目录下生成一个`fluid_inference_install_dir`目录，包含了预测库文件，后续用于配置此路径。

然后克隆此仓库
```
git clone https://github.com/Aurelius84/Inference_benchmark.git
cd paddle/inference
```

inference目录下有一个`image_classification.cc`，是resnet50/mobileNetv1的预测样例代码，可以编译测试：
```
apt-get update & apt install libprotobuf-dev
./re_build.sh
```

执行完毕后，会在当前目录生成一个`image_classification`可执行程序。如果本地有resnet的预测模型，则可以执行预测latency的评测：
```bash
./test_latency.sh 16 ../static/resnet50  

# 其中 16为batch_size, 后面为模型路径，可根据自己实际目录修改
```

#### 2. Torch环境搭建

Torch官方也提供了cuda10的镜像，首先拉取images:
```bash
docker pull pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
```

根据拉取的镜像，创建容器：
```bash
sudo nvidia-docker run --name XXX --net=host -v $PWD:/workspace -it pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel  /bin/bash
```

下载torch 1.6版本的官方预测库，并解压，会得到一个libtorch文件夹
```
apt-get update & apt-get install wget
wget https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcu101.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.6.0+cu101.zip
```

然后克隆此仓库，并将之前的libtorch文件夹放到torch目录下
```
apt-get update & apt-get install git -y
git clone https://github.com/Aurelius84/Inference_benchmark.git
cd torch
cp -r your/path/to/libtorch .
```

inference目录下有一个`image_classification.cpp`，是resnet50/mobileNetv1的预测样例代码，可以编译测试：
```
./re_build.sh
```

执行完毕后，会在当前目录生成一个`image_classification_exe`可执行程序。如果本地有resnet的预测模型，则可以执行预测latency的评测：
```bash
./test_latency.sh ../dy2stat/resnet50.pt

# 注意 torch的.pt模型需要用1.6.0版本的torch保存，路径可根据自己实际目录修改
```

### 3. Tensorflow环境搭建

Tensorflow并没有像Pytorch那样提供官方编译好的C++预测lib，需要依赖bazel进行编译，但由于环境复杂，编译流程较长，极容易踩坑。因此我们采用了开源的docker镜像。

下载支持cuda10.1的预编译好的镜像：
```bash
sudo nvidia-docker run --name XXX --net=host -v $PWD:/workspace -it floopcz/tensorflow_cc:ubuntu-cuda  /bin/bash
```

在根目录有一个`/tensorflow_cc`目录，是tensorflow的C++预测库项目

切换到`tensorflow/inference`目录，有一个`image_classification.cpp`样例文件，是resnet、mobilenet模型的预测代码
```
cd inference_benchmark/tensorflow/inference/
./re_build.sh  # 执行编译，生成image_classification执行文件

python image_classification.py # 保存模型

./image_classification  # 执行预测，load模型的路径是写死的目前（待优化）
```
