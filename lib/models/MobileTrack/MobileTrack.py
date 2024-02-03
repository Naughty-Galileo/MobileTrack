import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.layers.head import build_box_head
from lib.models.MobileTrack.LightHead_mobile import LightHead_mobile
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from lib.models.MobileTrack.connecttion import Point_Neck, Point_Neck_CoordAttention, Point_Neck_LightTrack, Point_Neck_FEAR, Hierarchy_Point_Neck
from lib.models.MobileTrack.MobileViT import MobileViT_S

class MLP_mixer(nn.Module):
    """ mixer, modeling channel, spatial """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, hw= 32*32):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers_channle = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layers_spatial = nn.ModuleList(nn.Linear(n, k) for n, k in zip([hw] + h, h + [hw]))

    def forward(self, x):
        for i in range(self.num_layers):
            x = x.permute(0, 1, 3, 2)
            x = F.relu(self.layers_spatial[i](x))
            x = x.permute(0, 1, 3, 2)
            x = F.relu(self.layers_channle[i](x)) if i < self.num_layers - 1 else self.layers_channle[i](x)
        return x

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MobileTrack(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        
        self.backbone = MobileViT_S()
        
        self.feature_fusor = Point_Neck_CoordAttention(num_kernel=64, cat=True, matrix=True,
                                                adjust=True, corr_channel=cfg.MODEL.CORRELATION_CHANNELS, adj_channel=cfg.MODEL.HEAD.NUM_CHANNELS)
        
        self.bbox_head = LightHead_mobile(cfg.MODEL.HEAD.NUM_CHANNELS)

        self.head_type = cfg.MODEL.HEAD.TYPE
    
        self.apply(self._init_weights)
        
        if cfg.MODEL.BACKBONE.PRETRAIN_FILE != "":
            def remove_prefix(state_dict, prefix):
                ''' Old style model is stored with all names of parameters
                share common prefix 'module.' '''
                f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
                return {f(key): value for key, value in state_dict.items()}

            def load_pretrain(model, pretrained_path):
                device = torch.cuda.current_device()
                pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
                pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.') 
                model.load_state_dict(pretrained_dict, strict=False)
                return model
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pretrained_path = os.path.join(current_dir, '../../../pretrained_models', cfg.MODEL.BACKBONE.PRETRAIN_FILE)
            load_pretrain(self.backbone, pretrained_path)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, z=None):
        zf = self.backbone(z)
        xf = self.backbone(x)
        
        feature = self.feature_fusor(zf, xf)
        B, C, H, W = feature.shape
        score_map_ctr, bbox, size_map, offset_map = self.bbox_head(feature, None)
        outputs_coord = bbox
        outputs_coord_new = outputs_coord.view(B, 1, 4)
        out = {'pred_boxes': outputs_coord_new,
                'score_map': score_map_ctr,
                'size_map': size_map,
                'offset_map': offset_map}
        return out

    def track(self, x):
        with torch.no_grad():
                xf = self.backbone(x)
                zf = self.z
                feature = self.feature_fusor(zf, xf)
                B, C, H, W = feature.shape
                score_map_ctr, bbox, size_map, offset_map = self.bbox_head(feature, None)
                outputs_coord = bbox
                outputs_coord_new = outputs_coord.view(B, 1, 4)
                out = {'pred_boxes': outputs_coord_new,
                       'score_map': score_map_ctr,
                       'size_map': size_map,
                       'offset_map': offset_map}
                return out
    
    def template(self, z):
        with torch.no_grad():
            z = self.backbone(z)
        self.z = z
        return z

    def template_z(self, z):
        self.z = z