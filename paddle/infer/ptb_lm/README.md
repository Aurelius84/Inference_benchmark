# Ptb Language model 测试

## 一、模型导出
### 1. 静态图模型
Ptb LM静态图模型以models的实现为准，模型目录在：[models/PaddleNLP/shared_modules/models/language_model/lm_model.py]()中。

在导入模型时，在`lm_model.py`的同级目录下，新建一个py文件，核心代码如下：

```python
import paddle.fluid as fluid
from lm_model import lm_model

import numpy as np

def save_inference_model():
    # define model
    res_vars = lm_model(
        hidden_size=200,
        vocab_size=10000,
        num_layers=2,
        num_steps=20,
        init_scale=0.1,
        dropout=None,
        rnn_model='static')
    proj, hid, last, feed_list = res_vars
    
    # modify into your model path
    paddle_model_dir = '/workspace/code_dev/paddle-predict/paddle/static/'

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()
    # run startup to initialize parameters
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(startup_program)
    # save model
    fluid.io.save_inference_model(dirname=paddle_model_dir+'ptb_lm',
                                feeded_var_names = feed_list,
                                target_vars = [proj, hid, last],
                                executor=exe,
                                main_program=main_program,
                                model_filename='model',
                                params_filename='params')

if __name__ == '__main__':
    save_inference_model()
```

**Note:**

1. `lm_model.py`中包括多种实现，目前使用的是`static`模式
2. `basic_lstm`模式的代码执行有bug，待确认问题。
3. 保持了`init_hidden`和`init_cell`的数据布局统一为[num_layer, batch_size, hidden_size]，因此需要移除掉多余的transpose和reshape层

### 2. 动转静模型

动转静模型原始代码是在models仓库中，之前在实现`to_static`功能，作为精度对齐，此处基于[paddle/python/paddle/fluid/test/unittests/dygraph_to_static/test_ptb_lm.py]()代码进行模型的导出。

在`test_ptb_lm.py`同级目录下，新建一个py文件，核心代码如下：

```python
import paddle
from test_ptb_lm import PtbModel

def save_inference_model():
    model = PtbModel(
            hidden_size=200,
            vocab_size=10000,
            num_layers=2,
            num_steps=20,
            init_scale=0.1,
            dropout=None)
    init_hidden = paddle.to_tensor(np.zeros((2, 4, 200), dtype='float32'))
    init_cell = paddle.to_tensor(np.zeros((2, 4, 200), dtype='float32'))
    input = paddle.to_tensor(np.arange(80).reshape(4, 20).astype('int64'))

    out, hid, last = model(input, init_hidden, init_cell)
    print(out.shape)
    
    paddle_model_dir = '/workspace/code_dev/paddle-predict/paddle/dy2stat/'
    config = paddle.SaveLoadConfig()
    config.model_filename = 'model'
    config.params_filename = 'params'
    config.output_spec = [out]
    paddle.jit.save(model, model_path=paddle_model_dir+'ptb_lm',input_spec=[input, init_hidden, init_cell], configs=config)


if __name__ == '__main__':
    save_inference_model()
```

**Note:**

1. 预测模型是不需要`label`输入的，因此需要移除`forward`函数中`label`输入
2. 预测模型不需要`loss`，因此也不需要loss相关的layer，因此需要移除

## 二、预测代码
详见`ptb_lm.cc`

## 三、性能评估
执行`./re_build.sh`脚本，对`ptb_lm.cc`执行编译，在当前目录下会生成一个`ptb_lm`可执行文件。

静态图模型预测：
```bash
./ptb_lm --use_gpu --batch_size=1 --repeat_time=100 --dirname=your_static_seq2seq_model_path
```

动转静模型预测：
```bash
./ptb_lm --use_gpu --batch_size=1 --repeat_time=100 --dirname=your_dy2stat_seq2seq_model_path
```