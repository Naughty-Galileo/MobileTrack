import math
import os

# for debug
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf
from lib.models.MobileTrack.MobileTrack import MobileTrack
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.stark_utils import PreprocessorX
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from lib.test.utils.hann import hann2d

class MobileTracker(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MobileTracker, self).__init__(params)
        network = MobileTrack(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorX()
        self.state = None
        
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        self.H, self.W, _ = image.shape
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template, _ = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = self.network.template(template)
        
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        self.frame_id += 1
        H, W, _ = image.shape

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_crop, _ = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # track
        with torch.no_grad():
            outputs = self.network.track(x_crop)
        
            # add hann windows
            pred_score_map = outputs['score_map']
            
            response = self.output_window * pred_score_map
            pred_boxes = self.network.bbox_head.cal_bbox(response, outputs['size_map'], outputs['offset_map'])
            pred_boxes = pred_boxes.view(-1, 4)
            
#             pred_boxes = outputs['pred_boxes'].view(-1, 4)

            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()

            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
            x, y, width, height = self.state

            self.center_pos = np.array([x+width/2, y+height/2])
            self.size = np.array([width, height])
            
            out = {'target_bbox': self.state }
            
            return out


    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights



def get_tracker_class():
    return MobileTracker
