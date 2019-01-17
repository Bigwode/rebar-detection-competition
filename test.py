import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils import array_tool as at
import pandas as pd

name, pred = [], []
id_list_file = '/home/csce/czwhhh/competition/rebar/VOC_GJ/ImageSets/Main/test.txt'

ids = [id_.strip() for id_ in open(id_list_file)]

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

trainer.load('./checkpoints/fasterrcnn_01151508_0')
opt.caffe_pretrain=True

for id in ids:
    print(id)
    img = read_image('/home/csce/czwhhh/competition/rebar/VOC_GJ/JPEGImages/{0}.jpg'.format(id))
    img = t.from_numpy(img)[None]
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
    all_box = at.tonumpy(_bboxes[0])

    for i in range(all_box.shape[0]):
        bbox = []
        bbox.append(str(int(round(all_box[i][1]))))
        bbox.append(str(int(round(all_box[i][0]))))
        bbox.append(str(int(round(all_box[i][3]))))
        bbox.append(str(int(round(all_box[i][2]))))
        name.append('{0}.jpg'.format(id))
        pred.append(' '.join(bbox))

submission = pd.DataFrame({'filename': name, 'label': pred})
submission.to_csv('submission.csv', header=None, index=False)

