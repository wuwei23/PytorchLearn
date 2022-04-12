import torch
from torch.autograd import Function
from PytorchLearn.Chapter9.SSD.layers.box_utils import decode, nms
from PytorchLearn.Chapter9.SSD.data import voc as cfg


class Detect(Function):
    # 测试阶段的最后一层, 负责解码预测结果, 应用nms选出合适的框和对应类别的置信度.
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh #阈值
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self,loc_data, conf_data, prior_data):
        """
        # loc_data: [batch, num_priors, 4], [batch, 8732, 4]
        # conf_data: [batch, num_priors, 21], [batch, 8732, 21]
        # prior_data: [num_priors, 4], [8732, 4]
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1) # 维度调换

        # Decode predictions into bboxes.
        for i in range(num):
            # 解码loc
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)#解码获取box坐标
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()# 复制第i个image置信度预测结果

            for cl in range(1, self.num_classes): #VOC又20个类+背景 = 21
                c_mask = conf_scores[cl].gt(self.conf_thresh)# 返回由0,1组成的数组, 0代表小于thresh, 1代表大于thresh
                scores = conf_scores[cl][c_mask]# 返回值为1的对应下标的元素值(即返回conf_scores中大于thresh的元素集合)
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes) # 获取对应box的二值矩阵
                boxes = decoded_boxes[l_mask].view(-1, 4)# 获取置信度大于thresh的box的左上角和右下角坐标

                # idx of highest scoring and non-overlapping boxes per class
                # 返回每个类别的最高的score 的下标, 并且除去那些与该box有较大交并比的box
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)# 从这些box里面选出top_k个, count<=top_k
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        # 注意, view共享tensor, 因此, 对flt的修改也会反应到output上面
        return output


if __name__ == "__main__":
    detect = Detect(21, 0, 200, 0.01, 0.5)
    loc_data = torch.randn(1,8732,4)
    conf_data = torch.randn(1,8732,21)
    prior_data = torch.randn(8732, 4)

    out = detect.forward(loc_data, conf_data, prior_data)
    print('Detect output shape:', out.shape)