from PIL import Image
import numpy as np
import torch

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

img = Image.open('catdog.png')
w, h = img.size
print(w, h)

d2l.set_figsize()


def display_anchors(fmap_w, fmap_h, s):
    # 前两维的取值不影响输出结果(原书这里是(1, 10, fmap_w, fmap_h), 我认为错了)
    fmap = torch.zeros((1, 10, fmap_h, fmap_w), dtype=torch.float32)

    # 平移所有锚框使均匀分布在图片上
    offset_x, offset_y = 1.0 / fmap_w, 1.0 / fmap_h
    anchors = d2l.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5]) + \
              torch.tensor([offset_x / 2, offset_y / 2, offset_x / 2, offset_y / 2])

    bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)

#小目标检测
display_anchors(fmap_w=4, fmap_h=2, s=[0.15])
d2l.plt.show()
#将特征图的高和宽分别减半，并用更大的锚框检测更大的目标
display_anchors(fmap_w=2, fmap_h=1, s=[0.4])
d2l.plt.show()