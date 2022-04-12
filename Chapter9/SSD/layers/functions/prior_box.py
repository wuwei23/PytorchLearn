from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    # priorbox实际上就是网格中每一个cell推荐的box
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        # 在SSD的init中, cfg=(coco, voc)[num_classes=21]
        # coco, voc的相关配置都来自于data/cfg.py 文件
        """
        1、计算先验框，根据feature map的每个像素生成box;
        2、框中个数为： 38×38×4+19×19×6+10×10×6+5×5×6+3×3×4+1×1×4=8732
        3、 cfg: SSD的参数配置，字典类型
        """
        super(PriorBox, self).__init__()
        self.image_size = cfg['image_size']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]  #方差
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']#比率
        self.clip = cfg['clip']#裁剪图像
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []#用来存放 box的参数（中心位置坐标x,y和宽高）
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                # 一个像素点生成好几个锚框
                # product 双层循环 i，j
                # k-th 层的feature map 大小， 一个图像分成steps[k]个特征图
                f_k = self.image_size / self.steps[k]
                # unit center x,y,求得center的坐标, 浮点类型.
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k #cy对应的是行, 所以应该 cy与i对应.

                # aspect_ratio: 1 时对应的box
                # rel size: min_size，r==1, size = s_k， 正方形
                s_k = self.min_sizes[k] / self.image_size#先验框大小相对于图片的比例
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1，当 aspect_ratios 为1时, 会有一个额外的 box, 如下:
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                # 其余(2, 或 2,3)的宽高比(aspect ratio)
                # 不为 1 时，产生的框为矩形
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        # 归一化，把输出设置在[0,1]
        if self.clip:
            output.clamp_(max=1, min=0)# clamp_ 是clamp的原地执行版本
        return output  # 输出default box坐标


if __name__ == "__main__":
    # SSD300 CONFIGS
    voc = {
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'image_size': 300,
        'steps': [8, 16, 32, 64, 100, 300],#38*8约等于300,所以第一个是8
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'VOC',
    }
    box = PriorBox(voc)
    print('Priors box shape:', box.forward().shape)
    print('Priors box:\n',box.forward())