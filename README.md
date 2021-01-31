# Inference_benchmark

### 一、测试环境：

+ Paddle：官方cuda10镜像，2.0beta分支
+ Torch：官方cuda10镜像，1.6.0版本
+ TensorFlow：开源cuda10.1镜像，2.2.0版本
+ 机器：P40，C++预测耗时

### [二、环境准备]((./tools/docker/README.md))

1. [Paddle 环境搭建](./tools/docker/paddle.sh)

2. [Torch 环境搭建](./tools/docker/torch.sh)

3. [Tensorflow 环境搭建](./tools/docker/tensorflow.sh)

### [三、模型准备](./tools/model/README.md)

1. [静态图预测模型](./tools/model/)

2. [动转静预测模型](./tools/model/fetch_model.sh)


### 四、预测性能评估

#### 1. Paddle 预测接口
paddle的预测接口开发，可以参考`paddle/inference/image_classification/image_classification.cc`中的代码。

核心代码如下：
```cpp
auto input_names = predictor->GetInputNames();
auto input_t = predictor->GetInputTensor(input_names[0]);
input_t->Reshape({batch_size, channels, height, width});
input_t->copy_from_cpu(input);
```

#### 2. Torch 预测接口
torch的预测接口开发更加简单，可以参考`torch/inference/image_classification.cpp`中的代码。

可以直接copy一份，修改一下`std::vector`的inputs中的数据即可：
```cpp
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::ones({batch_size, 3, 224,224}).to(device));
```

#### 3. 编译可执行文件
此步需要修改CMakeLists.txt，在`foreach`循环的文件中，添加新增的文件，会编译生成一个`your_new_file_exe`的可执行文件。
```
set(PredictorSRCFiles "image_classification.cpp"; "your_new_file.cpp")
```

#### 4. 测试预测latency
执行可执行文件，load模型，会输出预测的时间。

### 五、 参考文档
动态图转静态图文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/guides/dygraph_to_static/program_translator_cn.html


### 六、测试结果：

| 模型(单位：ms)  | bs |paddle（静） | paddle（动转静） |  torch   |  TF  |
| :-------------: | :----------: |:----------: | :----------: | :------: | :--: |
|  **ResNet50**   |      1       |   **5.94**  |     5.95     |   9.64   |   7.09   |
|                 |      4       |   **11.48** |     11.50    |   11.65  |   12.28  |
|                 |      16      |     35.09   |     34.98    |   34.48  |  **34.22**|
|                 |      32      |     72.62   |    **72.53** |    72.59 |  **64.30**|



| 模型(单位：ms) |  bs  | paddle（静） | paddle（动转静） | torch  |  TF  |
| :------------: | :--: | :----------: | :----------: | :----: | :--: |
| **ResNet101**  |  1   |  **10.96**   |    10.99     | 18.52  |   13.25   |
|                |  4   |  **18.88**   |    18.88     | 19.92  |   21.49   |
|                |  16  |  **55.60**   |    55.60     | 55.70  |   60.37   |
|                |  32  |  **119.31**  |    119.33    | 121.66 | **115.53**|



| 模型(单位：ms)  |  bs  | paddle（静） | paddle（动转静） | torch |  TF  |
| :-------------: | :--: | :----------: | :----------: | :---: | :--: |
| **MobileNetV1** |  1   |     1.77     |   **1.73**   | 3.39  | 2.62 |
|                 |  4   |   **3.14**   |     3.14     | 3.51  | 4.94 |
|                 |  16  |    11.31     |  **11.30**   | 11.95 | 16.79|
|                 |  32  |  **22.33**   |    22.33     | 23.79 | 31.72|


| 模型(单位：ms)  |  bs  | paddle（静） | paddle（动转静） | torch |  TF  |
| :-------------: | :--: | :----------: | :----------: | :---: | :--: |
| **YoloV3**      |  1   |   **11.92**  |     12.05    |   -   | 15.29|
|                 |  4   |   **21.48**  |     21.71    |   -   | 23.77|
|                 |  16  |   **60.42**  |     60.54    |   -   | 65.59|
|                 |  32  |     126.33   |     126.25   |    -  | **119.83** |

| 模型(单位：ms)  |  bs  | paddle（静） | paddle（动转静） | torch |  TF  |
| :-------------: | :--: | :----------: | :----------: | :---: | :--: |
| **ptb_lm**      |  1   |  6.75   |     6.85    |   -   | -|
|                 |  4   |   9.46  |     9.59    |   -   | - |
|                 |  16  |   19.37 |     19.49    |   7.63   | -|
|                 |  32  |   32.91 |     32.95   |    -  | - |


| 模型(单位：ms)  |  bs  | paddle（静） | paddle（动转静） | paddle(+break优化) |  TF  |
| :-------------: | :--: | :----------: | :----------: | :---: | :--: |
| **Seq2seq**     |  1   |  179.11   |     337.66    |   130.73   | -|
|                 |  4   |   188.78  |     349.98    |   143.98   | - |
|                 |  16  |   238.50  |     400.70    |   207.72   | -|
|                 |  32  |   327.72  |     494.46    |    295.61  | - |
