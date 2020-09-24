# Image Classification 测试

Image Classification(图片分类)领域选取了三个经典的模型：ResNet50、ResNet101、MobileNetV1。

## 一、模型导出

### 1. 静态图模型
ResNet50、ResNet101、MobileNetV1三个模型的静态图实现，均以models仓库代码为准，模型目录在：[models/PaddleCV/image_classification/models]()。

静态图模型导入只需要将对应的Model通过import导入，借助`fluid.io.save_inference_model`接口存储下来即可。在上述模型目录内创建一个py文件，包含如下核心代码：

```python
import paddle
import paddle.fluid as fluid

from resnet import ResNet50, ResNet101
from mobilenet_v1 import MobileNetV1
from mobilenet_v2 import MobileNetV2

# modify into your local model dirname
root_dir = '/workspace/all_models/'

def save_inference_model(model, model_name):
    main_prog = fluid.Program()
    start_prog = fluid.Program()
    with fluid.program_guard(main_prog, start_prog):
        # construct program
        img = fluid.data(shape=[None, 3, 224, 224], name='img')
        model = model()
        # align with dygraph with class_dim=1000
        y = model.net(img, class_dim=1000) 
    
        exe = fluid.Executor(fluid.CPUPlace())
        # run startup program to initialize all parameters
        exe.run(start_prog)

        fluid.io.save_inference_model(
            dirname=root_dir+model_name, 
            feeded_var_names=['img'], 
            target_vars=y,
            executor=exe,
            main_program=main_prog,
            model_filename='model',
            params_filename='params')

if __name__ == '__main__':
    save_inference_model(ResNet50, 'resnet50')
    save_inference_model(ResNet101, 'resnet101')

    save_inference_model(MobileNetV1, 'mobilenetv1')
    save_inference_model(MobileNetV2, 'mobilenetv2')
```

执行上述Python文件，可以导出ResNet50、ResNet101、MobileNetV1等静态图模型。

### 2. 动转静模型
ResNet50、ResNet101、MobileNetV1三个模型均不包含依赖Tensor的控制流语句（if/while）。此处仍以models仓库中的代码为基础测试，ResNet模型目录在：[models/dygraph/resnet]()，MobileNet模型目录在：[models/dygraph/mobilenet]路径下。

三个模型的导出代码基本一致。首先在对应的模型目录下创建py文件，核心代码如下：

```python
import paddle
import paddle.fluid as fluid
from paddle.static import InputSpec
from paddle.jit import to_static

from train import ResNet

# from mobilenet_v1 import MobileNetV1
# from mobilenet_v2 import MobileNetV2

# modify into your local model dirname
root_dir = '/workspace/all_models/'

def save_resnet(model, model_name):
    with fluid.dygraph.guard(fluid.CPUPlace()):
        # Distinguish ResNet50 and ResNet101
        if '101' in model_name:
            net = model(101, class_dim=1000)
        else:
            net = model(class_dim=1000)
        # Apply conversion of dy2stat
        net.forward = to_static(net.forward, [InputSpec([None, 3, 224, 224], name='img')])
        # Save configuration
        config = paddle.jit.SaveLoadConfig()
        config.model_filename = 'model'
        config.params_filename = 'params'
        paddle.jit.save(net, model_path=root_dir + model_name, configs=config)

def save_mobilenet(model, model_name):
    with fluid.dygraph.guard(fluid.CPUPlace()):
        # align static program with class_dim=1000
        net = model(scale=1.0, class_dim=1000)
        net.forward = to_static(net.forward, [InputSpec([None, 3, 224, 224], name='img')])

        # Save configuration
        config = paddle.jit.SaveLoadConfig()
        config.model_filename = 'model'
        config.params_filename = 'params'
        paddle.jit.save(net, model_path=root_dir + model_name, configs=config)
    

if __name__ == '__main__':
    save_resnet(ResNet, 'resnet50')
    save_resnet(ResNet, 'resnet101')

    # save_mobilenet(MobileNetV1, 'mobilenetv1')
    # save_mobilenet(MobileNetV2, 'mobilenetv2')
```

**Note:**
1. ResNet的动态图模型静态图实现，在最后一层有所区别：
    + 静态图预测代码最后的一个FC是**没有**softmax激活的
    + 动态图预测代码最后的一个FC是**带有**softmax激活的
    + 因此需要注释掉动态图中最后一个FC的act参数，保证公平

## 二、预测代码
详见`image_classification.cc`

## 三、性能评估
执行`./re_build.sh`脚本，对`image_classification.cc`执行编译，在当前目录下会生成一个`image_classification`可执行文件。

静态图模型预测：
```bash
./image_classification --use_gpu --batch_size=1 --repeat_time=100 --dirname=your_static_seq2seq_model_path
```

动转静模型预测：
```bash
./image_classification --use_gpu --batch_size=1 --repeat_time=100 --dirname=your_dy2stat_seq2seq_model_path
```