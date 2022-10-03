import io
import os
import json
import pickle
import datetime
import logging
import copy
import random
import math
import sys
import argparse
import glob
import time
from pathlib import Path
import threading

import numpy as np
import cv2
from PIL import Image

from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

import torch,torchvision
from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.config import *
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import RotatedCOCOEvaluator,DatasetEvaluators, inference_on_dataset, coco_evaluation,DatasetEvaluator
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

import matplotlib.pyplot as plt

from helpers_rbbox import my_transform_instance_annotations, RotatedPredictor
from helpers_coco import register_coco_instances_rbbox, AvgScoreEvaluator

setup_logger()

import datetime
print("----------------------------------------------------------")
print('Begin at:   ', datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))



def mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with our own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    image, transforms = T.apply_transform_gens([T.Resize((800, 800))], image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
      my_transform_instance_annotations(obj, transforms, image.shape[:2])
      for obj in dataset_dict.pop("annotations")
      if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances_rotated(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [RotatedCOCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper)


def prepare_dataset(args):
    dataset_train = f"{args.experiment_name}_train"
    dataset_val = f"{args.experiment_name}_val"
    data_dir = '/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/'
    train_annotation_file = data_dir + f"/{args.experiment_name}_train{args.tag}_coco/train.json"
    train_img_dir = data_dir + f"/{args.experiment_name}_train/"
    val_annotation_file = data_dir + f"/{args.experiment_name}_val{args.tag}_coco/val.json"
    val_img_dir = data_dir + f"/{args.experiment_name}_val/"

    DatasetCatalog.clear()  # resets catalog, helps prevent errors from running cells multiple times
    register_coco_instances_rbbox(dataset_train, {},
                            train_annotation_file,
                            train_img_dir)
    register_coco_instances_rbbox(dataset_val, {},
                            val_annotation_file,
                            val_img_dir)
    print('Registered Datasets: ', list(DatasetCatalog.data.keys()))

    return dataset_train, dataset_val


def prepare_cfg(setting, dataset_train, dataset_val):
    cfg = get_cfg()

    if 'base_cfg_file' in setting.keys():
        cfg.merge_from_file(setting['base_cfg_file'])
    else:
        #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        #cfg.MODEL.WEIGHTS = '/usr/workspace/zhong2/Research/FeedOpt/Questek/AMPIS/models/model_final_f10217.pkl'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = '/usr/workspace/zhong2/Research/FeedOpt/Questek/DefualtModels/R-50.pkl'
        cfg.DATASETS.TRAIN = ([dataset_train])
#         cfg.DATASETS.TEST = ([dataset_train, dataset_val, ])
        cfg.DATASETS.TEST = ([])

        ### SOLVER
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

        ### RBBOX
        cfg.MODEL.MASK_ON = False
        cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
        cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
        if 'rpn_box_reg_weights' in setting.keys():
            cfg.MODEL.RPN.BBOX_REG_WEIGHTS = setting['rpn_box_reg_weights']
        else:
            cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1,1,1,1,1)
        cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0,30,60,90,120,150]]
        cfg.MODEL.ROI_HEADS.NAME = "RROIHeads"
        cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
        if 'roi_head_box_reg_weights' in setting.keys():
            cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = setting['roi_head_box_reg_weights']
        else:
            cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10,10,5,5,1)

        ### INPUT & BACKBONE
        if 'backbone_freeze_at' in setting.keys():
            cfg.MODEL.BACKBONE.FREEZE_AT = int(setting['backbone_freeze_at'])
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
#         if 'rpn_boundary_thresh' in setting.keys():
#             cfg.MODEL.RPN.BOUNDARY_THRESH = int(setting['rpn_boundary_thresh'])
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
        if 'anchor_shapes' in setting.keys():
            cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [float(shape) for shape in setting['anchor_shapes']]

        ### ROI_HEADS
#         if '' in setting.keys():
#             cfg.MODEL. = setting['']
#         cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
#         cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 4
#             cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = 8

        cfg.TEST.DETECTIONS_PER_IMAGE = 150
        cfg.TEST.EVAL_PERIOD = 200

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.SOLVER.MAX_ITER = 4000
    gpu_idx = setting['gpu_idx']
    cfg.MODEL.DEVICE = f'cuda:{gpu_idx}'
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
        experiment_id = hps_input_file.split('/')[-2]
        out_dir_prefix = f'/usr/workspace/zhong2/Research/FeedOpt/Questek/Results/Tuning_{args.experiment_name}/{experiment_id}/output{args.tag}/'
        setting_name = f'setting_{args.start_idx + i}/'
        out_dir = out_dir_prefix + setting_name
#     setting = {}
#     i = 0
#     out_dir = '/usr/workspace/zhong2/Research/FeedOpt/Questek/Results/nanorods/'
        setting.update({'out_dir': out_dir})
        setting.update({'gpu_idx': gpu_idx})

        cfg = prepare_cfg(setting, dataset_train, dataset_val)

            ###----------------------------------------------------------------
            ### Train
    #         if train:
        trainer = MyTrainer(cfg)  # create trainer object from cfg
        trainer.resume_or_load(resume=False)  # start training from iteration 0
        try:
            trainer.train()  # train the model!
        except:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(f'setting {i+args.start_idx} error!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            sys.stdout.flush()

#         if test_coco:
#             predictor = DefaultPredictor(cfg)
#             evaluator = COCOEvaluator("balloon_val", output_dir="./output")
#             val_loader = build_detection_test_loader(cfg, "balloon_val")
#                 print(inference_on_dataset(predictor.model, val_loader, evaluator))
            ### another equivalent way to evaluate the model is to use `trainer.test`

    return

def main(args, gpu_idx):
    dataset_train, dataset_val = prepare_dataset(args)
    train_model(args, dataset_train, dataset_val, gpu_idx)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train maskRCNN")
    parser.add_argument('--experiment_name', type=str, default='nanorods',
                        help='vesicles | nanorods')
    parser.add_argument('--hps_input0', type=str, default='',
                        help='a .pkl file containing hyper parameters to tune for one machine. Written by write_hps.py')
    parser.add_argument('--hps_input1', type=str, default='',
                        help='a .pkl file containing hyper parameters to tune for one machine. Written by write_hps.py')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='start_idx for continue to run a setting')
    parser.add_argument('--tag', type=str, default='',
                        help='extra information')
#     parser.add_argument('--num_gpus', type=int, default=2,
#                         help='num_gpus_per_machine')
#     parser.add_argument('--num_machines', type=int, default=1,
#                         help='the total number of machines')
#     parser.add_argument('--machine_rank', type=int, default=0,
#                         help='the rank of this machine')
    args = parser.parse_args()

#     main(args, 0)
    threads = []
    for gpu_idx in np.arange(2):
        threads.append(threading.Thread(target=main, args=(args, gpu_idx)))
        threads[-1].start()
        print(f'threds {gpu_idx} started')



time.sleep(1)
print('End at:   ', datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
print("----------------------------------------------------------")


