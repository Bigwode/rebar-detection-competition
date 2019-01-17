# rebar-detection-competition-baseline
Simple rebar detection competition baseline(0.97+) based on Faster RCNN[1]

competition link:  https://www.datafountain.cn/competitions/332/details

**If you wanna a higher score.Please use FPN or DCN，maybe softnms(also contained in this baseline) or other tricks are useful.Thanks for contribution!**

### 1、Install dependencies

requires PyTorch >=0.4

- install PyTorch >=0.4 with GPU (code are GPU-only), refer to [official website](http://pytorch.org)

- install cupy, you can install via `pip install cupy-cuda80` or(cupy-cuda90,cupy-cuda91, etc).

- install other dependencies:  `pip install -r requirements.txt `

- Optional, but strongly recommended: build cython code `nms_gpu_post`: 

  ```bash
  cd model/utils/nms/
  python build.py build_ext --inplace
  cd -
  ```

- start visdom for visualization

```bash
nohup python -m visdom.server &
```

### 2、 Prepare caffe-pretrained vgg16

First，**Download pretrained model** from [Google Drive](https://drive.google.com/open?id=1cQ27LIn-Rig4-Uayzy_gH5-cW-NRGVzY) or [Baidu Netdisk( passwd: scxn)](https://pan.baidu.com/s/1o87RuXW)

See [demo.ipynb](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/demo.ipynb) for more detail.

If you want to use caffe-pretrain model as initial weight, you can run below to get vgg16 weights converted from caffe, which is the same as the origin paper use.

```bash
python misc/convert_caffe_pretrain.py
```

This scripts would download pretrained model and converted it to the format compatible with torchvision. If you are in China and can not download the pretrain model, you may refer to [this issue](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/issues/63)

Then you could specify where caffe-pretraind model `vgg16_caffe.pth` stored in `utils/config.py` by setting `caffe_pretrain_path`. The default path is ok.

If you want to use pretrained model from torchvision, you may skip this step.

**NOTE**, caffe pretrained model has shown slight better performance.

**NOTE**: caffe model require images in BGR 0-255, while torchvision model requires images in RGB and 0-1. See `data/dataset.py`for more detail. 

### 3、Prepare your data.

refer to: https://blog.csdn.net/github_36923418/article/details/86303670

### 4、Train

```bash
python train.py train --env='fasterrcnn-caffe' --plot-every=100 --caffe-pretrain
```

you may refer to `utils/config.py` for more argument.

Some Key arguments:

- `--caffe-pretrain=False`: use pretrain model from caffe or torchvision (Default: torchvison)
- `--plot-every=n`: visualize prediction, loss etc every `n` batches.
- `--env`: visdom env for visualization
- `--voc_data_dir`: where the VOC data stored
- `--use-drop`: use dropout in RoI head, default False
- `--use-Adam`: use Adam instead of SGD, default SGD. (You need set a very low `lr` for Adam)
- `--load-path`: pretrained model path, default `None`, if it's specified, it would be loaded.


### 5、Test

```python
python test.py
```

**If you have any question about this baseline, please ask questions in the issue area, I will give you an answer as soon as possible.**

**reference**

[1]、https://arxiv.org/abs/1506.01497

[2]、https://github.com/chenyuntc/simple-faster-rcnn-pytorch