### 模型准备

测试paddle和竞品torch、tf的动转静预测性能，首先需要保存 **动转静** 后的模型。详细的模型导出教程参考各目录下readme.

整体流程上，paddle可以通过`@to_static`装饰`forward`函数，然后调用`jit.save`保存。
```python
import paddle
import paddle.fluid as fluid
from paddle.static import InputSpec
from paddle.jit import to_static


def save_paddle_resnet(model, model_name):
    with fluid.dygraph.guard(fluid.CPUPlace()):
        if '101' in model_name:
            net = model(101, class_dim=1000)
        else:
            net = model(class_dim=1000)
        net.forward = to_static(net.forward, [InputSpec([None, 3, 224, 224], name='img')])
        config = paddle.SaveLoadConfig()
        config.model_filename = 'model'
        config.params_filename = 'params'
        paddle.jit.save(net, model_path=paddle_model_dir + model_name, configs=config)

```

torch可以通过`torch.jit.script`处理model，然后调用`model.save(path)`即可保存`.pt`模型
```python
import torch
import torchvision.models as models

def save_torch_resnet101():
    resnet = models.resnet101(pretrained=False)
    resnet = torch.jit.script(resnet)
    resnet.save(torch_model_dir + "resnet101.pt")
```

目前需要测的模型列表是：resnet50、mobilenetV1、seq2seq、ptb_lm、yolov3

paddle和torch的模型实现，见仓库：https://github.com/phlrain/example


tensorflow的可以将keras的模型保存为一个单独的`.pb`文件，以供C++端直接加载，详细步骤请参考：[教程](https://blog.csdn.net/ouening/article/details/104335552)