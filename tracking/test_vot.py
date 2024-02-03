import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import importlib
from lib.utils.lmdb_utils import decode_img
from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker
from SOTDrawRect.toolkit.datasets import DatasetFactory
from lib.test.evaluation.environment import env_settings

def get_parameters(name, parameter_name):
    """Get parameters."""
    param_module = importlib.import_module('lib.test.parameter.{}'.format(name))
    params = param_module.parameters(parameter_name)
    return params


def save_bb(file, data):
    tracked_bb = np.array(data).astype(int)
    np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')
    
    
def save_results(tracker, output):
    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)
    
    if seq.dataset in ['trackingnet', 'got10k']:
        if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset)):
            os.makedirs(os.path.join(tracker.results_dir, seq.dataset))

    if seq.dataset in ['trackingnet', 'got10k']:
        base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
    else:
        base_results_path = os.path.join(tracker.results_dir, seq.name)
    
    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--run_id', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='VOT2019', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--datset_root', type=str, default='/root/autodl-tmp/vot2019', help='datset_root')
    args = parser.parse_args()

    dataset = DatasetFactory.create_dataset(name=args.dataset_name, dataset_root=args.datset_root, load_img=False)
    
    name = args.tracker_name
    parameter_name = args.tracker_param
    
    env = env_settings()
    results_dir = '{}/{}/{}'.format(env.results_path, name, parameter_name)
    
    tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','lib/test/tracker', '%s.py' % name))
    
    if os.path.isfile(tracker_module_abspath):
        tracker_module = importlib.import_module('lib.test.tracker.{}'.format(name))
        tracker_class = tracker_module.get_tracker_class()
    
    params = get_parameters(name, parameter_name)
    tracker = tracker_class(params, args.dataset_name)
    
    for video in dataset:
        for idx, (img, gt_bbox) in enumerate(video):
            out = {}
            if idx == 0:
                if len(gt_bbox) == 8:
                    x1, y1 = gt_bbox[0], gt_bbox[1]
                    x2, y2 = gt_bbox[2], gt_bbox[3]
                    x3, y3 = gt_bbox[4], gt_bbox[5]
                    x4, y4 = gt_bbox[6], gt_bbox[7]
                    xmin = min(x1, x2, x3, x4)
                    xmax = max(x1, x2, x3, x4)
                    ymin = min(y1, y2, y3, y4)
                    ymax = max(y1, y2, y3, y4)
                    w = xmax - xmin
                    h = ymax - ymin
                else:
                    xmin, ymin = gt_bbox[0], gt_bbox[1]
                    w, h = gt_bbox[2], gt_bbox[3]
                info = {'init_bbox' : [xmin, ymin, w, h]}
                tracker.initialize(img, info)
            else:
                out = tracker.track(img)
                
                
if __name__ == '__main__':
    main()
