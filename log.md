### 训练记录

1.12

修改config.py文件中voc_data_dir='/home/csce/czwhhh/competition/rebar/VOC_GJ/'

先不修改min_size和max_size，感觉调大点会比较好，但是又怕显存不够。

voc_dataset.py 修改split='train'

`VOC_BBOX_LABEL_NAMES`只有person;

faster_rcnn_vgg16.py修改n_fg_class=1

vis_tool.py VOC_BBOX_LABEL_NAMES=('person')

trainer.py修改`self.roi_cm = ConfusionMeter(2)`

需要做的事：统计bbox的大小，重新修改anchor的scale大小，统计anchor

------

error:

FileNotFoundError: [Errno 2] No such file or directory:

'/home/csce/czwhhh/competition/rebar/VOC_GJ/Annotations/D0BF9992.xml'

去掉testDataloader，因为没有验证集。

```
第一次提交得分：0.32770400
第二次，使用原图大小2660*2000，0.96755500
第三次 anchor_scales=[8, 16, 32]变成[4, 8, 18] 0.96063200000
```



```python
Conformable-CNN(Deprecated)
出现错误：16:49:37] src/operator/././cudnn_algoreg-inl.h:112: Running performance tests to find the best convolution algorithm, this can take a while... (setting env variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)
解决方案：使用mxnet-1.0.0版本

```



1、使用FPN

2、加softnms 0.67176

https://www.smwenku.com/a/5b85ab9d2b71775d1cd39cc2/zh-cn/

```python
# soft nms in faster_rcnn.py
det = np.concatenate((cls_bbox_l, prob_l.reshape(-1, 1)), axis=1)
keep = soft_nms(det)使用的是默认参数。
soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1): score_threshold可以设置为0.7
```

sigma=0.5,Nt=0.1, threshold=0.7, 0.92662300000

```
sigma=0.5, Nt=0.5, threshold=0.7, method=2
```



3、调大roi数量。修改creator_tool.py中ProposalCreator默认参数

```python
n_test_post_nms=350,  # 300
min_size=12  # 16
```

```python
n_test_pre_nms=12000,
n_test_post_nms=2000,  # 300
min_size=10  # 16
anchor_scales=[4, 8, 16, 32]
```

