import torch
import torch.nn as nn
from lib.models.MobileTrack.connecttion import CoordAttention

class Conv2dBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(Conv2dBNReLU, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.BN = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.ReLU(self.BN(x))
        return x

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

    
    
class cls_pred_head(nn.Module):
    def __init__(self, inchannels=256):
        super(cls_pred_head, self).__init__()
        self.cls_pred = nn.Conv2d(inchannels, 1, kernel_size=1) # , stride=1, padding=1

    def forward(self, x):
        # x = 0.1 * self.cls_pred(x)
        x = self.cls_pred(x)
        return x

class size_pred_head(nn.Module):
    def __init__(self, inchannels=256):
        super(size_pred_head, self).__init__()
        # reg head
        self.size_pred = nn.Conv2d(inchannels, 2, kernel_size=1) # kernel_size=3, stride=1, padding=1
        # adjust scale
#         self.adjust = nn.Parameter(0.1 * torch.ones(1))
#         self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)))

    def forward(self, x):
#         x = self.adjust * self.bbox_pred(x) + self.bias
#         x = torch.exp(x)
        x=self.size_pred(x)
        return x

class offset_pred_head(nn.Module):
    def __init__(self, inchannels=256):
        super(offset_pred_head, self).__init__()
        # reg head
        self.offset_pred = nn.Conv2d(inchannels, 2, kernel_size=1) # kernel_size=3, stride=1, padding=1
        # adjust scale
#         self.adjust = nn.Parameter(0.1 * torch.ones(1))
#         self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)))

    def forward(self, x):
#         x = self.adjust * self.bbox_pred(x) + self.bias
#         x = torch.exp(x)
        x = self.offset_pred(x)
        return x


class LightHead_mobile(nn.Module):
    def __init__(self, channel = 256, feat_sz=16, stride=8):
        super(LightHead_mobile, self).__init__()
        self.tower = nn.Sequential(
#             Conv2dBNReLU(channel, channel//2, kernel_size=3, stride=1, padding=1),
#             Conv2dBNReLU(channel//2, channel//4, kernel_size=3, stride=1, padding=1)
            conv(channel, channel, freeze_bn=False),
            conv(channel, channel//2, freeze_bn=False),
            conv(channel//2, channel//4, freeze_bn=False),
        )

        self.size = conv(channel//4, channel//8, freeze_bn=False)
        self.cls = conv(channel//4, channel//8, freeze_bn=False)
        self.offset = conv(channel//4, channel//8, freeze_bn=False)

        self.cls_pred = cls_pred_head(inchannels=channel//8)
        self.sz_pred = size_pred_head(inchannels=channel//8)
        self.offset_pred = offset_pred_head(inchannels=channel//8)
        
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, gt_score_map=None):
        x = self.tower(x)
        
        sz = self.size(x)
        clss = self.cls(x)
        offset = self.offset(x)
        
        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        score_map_ctr = _sigmoid(self.cls_pred(clss))  # B 1 16 16
        score_map_size = _sigmoid(self.sz_pred(sz))
        score_map_offset = self.offset_pred(offset)
        
#         bbox_map = self.bbox_pred(reg) # B 4 16 16
#         print(score_map_size.shape)
#         print(score_map_ctr.shape)
#         print(score_map_offset.shape)
        
        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, score_map_size, score_map_offset)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), score_map_size, score_map_offset)
        return score_map_ctr, bbox, score_map_size, score_map_offset


    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

#         idx = idx.unsqueeze(1).expand(idx.shape[0], 4, 1)
#         offset = bbox_map.flatten(2).gather(dim=2, index=idx)
#         xyxy = torch.cat([(idx_x.to(torch.float) - offset[:, 0]) / self.feat_sz,
#                           (idx_y.to(torch.float) - offset[:, 1]) / self.feat_sz,
#                           (idx_x.to(torch.float) + offset[:, 2]) / self.feat_sz,
#                           (idx_y.to(torch.float) + offset[:, 3]) / self.feat_sz,
#                          ], dim=1)
        
#         bbox = torch.cat([(idx_x.to(torch.float) + idx_x.to(torch.float) + offset[:, 2] - offset[:, 0]) / (2*self.feat_sz),
#                           (idx_y.to(torch.float) + idx_y.to(torch.float) + offset[:, 3] - offset[:, 1]) / (2*self.feat_sz),
#                           (offset[:, 0] + offset[:, 2]) / self.feat_sz,
#                           (offset[:, 1] + offset[:, 3]) / self.feat_sz,
#                          ], dim=1)
        
        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)
#         print(bbox)
        if return_score:
            return bbox, max_score
        return bbox
