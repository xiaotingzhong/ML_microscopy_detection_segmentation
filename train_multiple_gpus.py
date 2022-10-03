""" This script take in two settings_X.pkl file and runs settings in them using two GPUs """
import os
import sys
import argparse
import time
import threading
from pathlib import Path
import pickle
import json

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import cv2

## detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
)

from detectron2.config import CfgNode
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from helpers_defaults import BestTrainer
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)
from helpers_coco import AvgScoreEvaluator


import datetime
print("----------------------------------------------------------")
print('Begin at:   ', datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))



class Trainer(BestTrainer):
    """
    "DefaultTrainer" contains a number pre-defined logic for standard training workflow.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "val_res/")
        print('!!!!!!!!!!!!!!!!!! Hey doing evaluation !!!!!!!!!!!!!!!!!!')
        return COCOEvaluator(dataset_name, output_dir=output_folder)
#         return [COCOEvaluator(dataset_name, output_dir=output_folder),
#                 AvgScoreEvaluator(dataset_name, output_dir=output_folder)]

    ### NOTICE: doing segmentation will increase memory and training time about 4 times
#     @classmethod
#     def build_train_loader(cls, cfg):
#         mapper = DatasetMapper(cfg, is_train=True, augmentations=[
#                                 T.RandomRotation(angle=np.arange(4)*90, sample_style="choice"),
#                                 T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
#                                 T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
#                                 T.RandomContrast(0.8, 3),
#                                 T.RandomBrightness(0.8, 1.6),
#                                ])
#         dataloader = build_detection_train_loader(cfg, mapper=mapper)
#         print('!!!!!!!!!!!!!!!!!! Hey doing augmentation !!!!!!!!!!!!!!!!!!')
#         return dataloader


def prepare_dataset(args):
    dataset_train = f"{args.experiment_name}_train"
    dataset_val = f"{args.experiment_name}_val"
    data_dir = '/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/'
#     train_annotation_file = data_dir + f"/{args.experiment_name}_train_coco_{args.color}{args.tag}/train.json"
#     train_img_dir = data_dir + f"/{args.experiment_name}_train/"
    train_annotation_file = data_dir + f"/{args.experiment_name}_train_coco_{args.color}/train.json"
    train_img_dir = data_dir + f"/{args.experiment_name}_train/"
    val_annotation_file = data_dir + f"/{args.experiment_name}_val_coco_{args.color}/val.json"
    val_img_dir = data_dir + f"/{args.experiment_name}_val/"

    DatasetCatalog.clear()  # resets catalog, helps prevent errors from running cells multiple times
    register_coco_instances(dataset_train, {},
                            train_annotation_file,
                            train_img_dir)
    register_coco_instances(dataset_val, {},
                            val_annotation_file,
                            val_img_dir)
    print('Registered Datasets: ', list(DatasetCatalog.data.keys()))
    return dataset_train, dataset_val

def prepare_cfg(setting, dataset_train, dataset_val):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))

    cfg.INPUT.MASK_FORMAT = 'polygon'
#     cfg.INPUT.FORMAT = 'RGB'
    cfg.DATASETS.TRAIN = (dataset_train,)
#     cfg.DATASETS.TEST = (dataset_train, dataset_val, )
    cfg.DATASETS.TEST = ()

    cfg.SOLVER.CHECKPOINT_PERIOD = 3000
    cfg.SOLVER.MAX_ITER = 2000
    if 'base_lr' in setting.keys():
        cfg.SOLVER.BASE_LR = float(setting['base_lr'])
    if 'imgs_per_batch' in setting.keys():
        cfg.SOLVER.IMS_PER_BATCH = int(setting['imgs_per_batch'])
    if 'gamma' in setting.keys():
        cfg.SOLVER.GAMMA = float(setting['gamma'])
    if 'nestrov' in setting.keys():
        cfg.SOLVER.NESTEROV = bool(setting['nestrov'])
    if 'steps' in setting.keys():
        cfg.SOLVER.STEPS = tuple([int(step) for step in setting['steps']])

#     cfg.SOLVER.BEST_CHECKPOINTER = CfgNode({"ENABLED": False})
#     cfg.SOLVER.BEST_CHECKPOINTER.METRIC = f"{dataset_val}/bbox/AP"
#     cfg.SOLVER.BEST_CHECKPOINTER.MODE = "max"
#     ### need to change DefaultTrainer to BestTrainer


    ###  Input and FPN
    if 'fpn_feature_norm' in setting.keys():
        cfg.MODEL.FPN.NORM = str(setting['fpn_feature_norm'])
    if 'input_min_size_train' in setting.keys():
        cfg.INPUT.MIN_SIZE_TRAIN = int(setting['input_min_size_train'])
    if 'input_max_size_train' in setting.keys():
        cfg.INPUT.MAX_SIZE_TRAIN = int(setting['input_max_size_train'])
    if 'input_min_size_test' in setting.keys():
        cfg.INPUT.MIN_SIZE_TEST = int(setting['input_min_size_test'])
    if 'input_max_size_test' in setting.keys():
        cfg.INPUT.MAX_SIZE_TEST = int(setting['input_max_size_test'])

    ### RPN
    if 'rpn_in_features' in setting.keys():
        cfg.MODEL.RPN.IN_FEATURES = [str(item) for item in setting['rpn_in_features']]
    if 'rpn_boundary_thresh' in setting.keys():
        cfg.MODEL.RPN.BOUNDARY_THRESH = int(setting['rpn_boundary_thresh'])
    if 'rpn_iou_thresh' in setting.keys():
        cfg.MODEL.RPN.IOU_THRESHOLDS = [float(thresh) for thresh in setting['rpn_iou_thresh']]
    if 'rpn_batch_per_img' in setting.keys():
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = int(setting['rpn_batch_per_img'])
    if 'rpn_smooth_l1_beta' in setting.keys():
        cfg.MODEL.RPN.SMOOTH_L1_BETA = float(setting['rpn_smooth_l1_beta'])
    if 'rpn_pre_nms_topk_train' in setting.keys():
        cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = int(setting['rpn_pre_nms_topk_train'])
    if 'rpn_pre_nms_topk_test' in setting.keys():
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = int(setting['rpn_pre_nms_topk_test'])
    if 'rpn_post_nms_topk_train' in setting.keys():
        cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = int(setting['rpn_post_nms_topk_train'])
    if 'rpn_post_nms_topk_test' in setting.keys():
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = int(setting['rpn_post_nms_topk_test'])
    if 'rpn_nms_thresh' in setting.keys():
        cfg.MODEL.RPN.NMS_THRESH = float(setting['rpn_nms_thresh'])
    if 'anchor_sizes' in setting.keys():
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[int(size[0])] for size in setting['anchor_sizes']]


    cfg.MODEL.ROI_HEADS.NUM_CLASSES = setting['n_classes']
    gpu_idx = setting['gpu_idx']
    cfg.MODEL.DEVICE = f'cuda:{gpu_idx}'
    cfg.MODEL.WEIGHTS = str('/usr/workspace/zhong2/Research/FeedOpt/Questek/AMPIS/models/model_final_f10217.pkl')

    cfg.TEST.EVAL_PERIOD = 500
    if args.color == '2':
        cfg.TEST.DETECTIONS_PER_IMAGE = 500 # maximum #instances that can be detected in an image is fixed in mask r-cnn)
    else:
        cfg.TEST.DETECTIONS_PER_IMAGE = 150

#     setting_name = f'nclasses_{n_classes}_'
#     for key in sorted(setting.keys()):
#         setting_name += key + '_' + str(setting[key]) + '_'
#     out_dir = out_dir_prefix + setting_name[:-1] + '/'
    cfg.OUTPUT_DIR = setting['out_dir']
    os.makedirs(Path(cfg.OUTPUT_DIR), exist_ok=True)

    with open(cfg.OUTPUT_DIR+"config.yaml", "w") as f:
        f.write(cfg.dump())   # save config to file

    return cfg

def train_model(args, dataset_train, dataset_val, gpu_idx, train=True, test_coco=False):
    if gpu_idx == 0:
        hps_input_file = args.hps_input0
    else:
        hps_input_file = args.hps_input1
    with open(hps_input_file, 'rb') as f:
        hp_settings = pickle.load(f)

    for (i, setting) in enumerate(hp_settings[args.start_idx:]):
        ###----------------------------------------------------------------
        ### Model Config
#         n_classes = int(gpu_idx + 1)
#         n_classes = int(np.random.choice([1, 2]))
        n_classes = 2
        if args.color == 'all':
            n_classes = 5
        setting.update({'n_classes': n_classes})

        experiment_id = hps_input_file.split('/')[-2]
        out_dir_prefix = f'/usr/workspace/zhong2/Research/FeedOpt/Questek/Results/Tuning_{args.experiment_name}/{experiment_id}/output_{args.color}{args.tag}/'
        setting_name = f'nclasses_{n_classes}_setting_{args.start_idx + i}/'
        out_dir = out_dir_prefix + setting_name
        setting.update({'out_dir': out_dir})
        setting.update({'gpu_idx': gpu_idx})

        cfg = prepare_cfg(setting, dataset_train, dataset_val)

        ###----------------------------------------------------------------
        ### Train
#         trainer = Trainer(cfg)  # create trainer object from cfg
        trainer = DefaultTrainer(cfg)  # create trainer object from cfg
        trainer.resume_or_load(resume=False)  # start training from iteration 0
        try:
            trainer.train()  # train the model!
        except:
            print('!!!!!2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(f'setting {i+args.start_idx} error!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            sys.stdout.flush()

    return

def main(args, gpu_idx):
    dataset_train, dataset_val = prepare_dataset(args)
    train_model(args, dataset_train, dataset_val, gpu_idx)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train maskRCNN")
    parser.add_argument('--experiment_name', type=str, default='vesicles',
                        help='vesicles | nanorods')
    parser.add_argument('--color', type=str, default='all',
                        help='if a number is given, train model for targets from a single category; if all, train model for all categories. Idea form Holm group paper.')
    parser.add_argument('--hps_input0', type=str, default='',
                        help='a .pkl file containing hyper parameters to tune for one machine. Written by write_hps.py')
    parser.add_argument('--hps_input1', type=str, default='',
                        help='a .pkl file containing hyper parameters to tune for one machine. Written by write_hps.py')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='start_idx for continue to run a setting')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for output dir name')
#     parser.add_argument('--num_gpus', type=int, default=2,
#                         help='num_gpus_per_machine')
#     parser.add_argument('--num_machines', type=int, default=1,
#                         help='the total number of machines')
#     parser.add_argument('--machine_rank', type=int, default=0,
#                         help='the rank of this machine')
    args = parser.parse_args()

    if len(args.hps_input1) == 0:
        main(args, 0)
    else:
        threads = []
        for gpu_idx in np.arange(2):
            threads.append(threading.Thread(target=main, args=(args, gpu_idx)))
            threads[-1].start()
            print(f'threds {gpu_idx} started')


#     launch(  # https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/launch.html
#         main,
#         args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=None,
#         args=(args,),
#     )

time.sleep(1)
print('End at:   ', datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
print("----------------------------------------------------------")

