"""
This script takes in two files containing lists of model weight files,
give these weights to models and evaluate models using two GPUs.
"""

import detectron2
import contextlib
import datetime
import io
import os
import sys
import threading
import pickle
import json
import logging
import cv2
import random
import numpy as np
import copy,torch,torchvision
import PIL
from PIL import Image
import xml.etree.ElementTree as X
import math
from itertools import repeat
import glob
import time
import shutil
import argparse
from multiprocessing.pool import ThreadPool
import concurrent.futures
import torch

from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

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


# torch.cuda.set_device(0)

from helpers_coco import register_coco_instances_rbbox

setup_logger()


def my_transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
    if annotation["bbox_mode"] == BoxMode.XYWHA_ABS:
        annotation["bbox"] = transforms.apply_rotated_box(np.asarray([annotation["bbox"]]))[0]
    else:
        bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # Note that bbox is 1d (per-instance bounding box)
        annotation["bbox"] = transforms.apply_box([bbox])[0]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

    return annotation

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

# class MyRotatedCOCOEvaluator(RotatedCOCOEvaluator):
#     def _eval_predictions(self, tasks, predictions, img_ids=None):
#         super()._eval_predictions(tasks, predictions)


def prepare_datasets(EXPERIMENT_NAME='nanorods'):
    data_dir = '/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/'
    DatasetCatalog.clear()  # resets catalog, helps prevent errors from running cells multiple times
    for tag in ['',  '_2classes',  '_noOverlap']:
        dataset_train = f"{EXPERIMENT_NAME}_train{tag}"
        dataset_val = f"{EXPERIMENT_NAME}_val{tag}"
        train_annotation_file = data_dir + f"/{EXPERIMENT_NAME}_train{tag}_coco/train.json"
        train_img_dir = data_dir + f"/{EXPERIMENT_NAME}_train/"
        val_annotation_file = data_dir + f"/{EXPERIMENT_NAME}_val{tag}_coco/val.json"
        val_img_dir = data_dir + f"/{EXPERIMENT_NAME}_val/"

        register_coco_instances_rbbox(dataset_train, {},
                                        train_annotation_file,
                                        train_img_dir)
        register_coco_instances_rbbox(dataset_val, {},
                                        val_annotation_file,
                                        val_img_dir)
    print('Registered Datasets: ', list(DatasetCatalog.data.keys()))
    return

def eval_model(model_weights_file, cfg_file, dataset_train, dataset_val, gpu_idx=0):
    ### Prepare predictor model
    cfg = get_cfg()
    try:  # load parameters
        cfg.merge_from_file(cfg_file)
    except:   # avoid type mismatch error
        cfg.INPUT.MIN_SIZE_TRAIN = 0
        cfg.merge_from_file(cfg_file)
    res_dir = cfg.OUTPUT_DIR
    model_name = model_weights_file.split('/')[-1].split('.')[0]

    cfg.MODEL.WEIGHTS = model_weights_file  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    cfg.MODEL.DEVICE = f'cuda:{gpu_idx}'

    ### Evaluate with predictor
    res = {}
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    if dataset_train:
        output_dir = res_dir+f"{model_name}_output_train_again"
        os.makedirs(output_dir, exist_ok=True)

        evaluator = RotatedCOCOEvaluator(dataset_train, cfg, False, output_dir=output_dir)
        loader = build_detection_test_loader(cfg, dataset_train, mapper=mapper)
        summary_train = inference_on_dataset(trainer.model, loader, evaluator)

        res.update({dataset_train: summary_train})

    if dataset_val:
        output_dir = res_dir+f"{model_name}_output_val_again"
        os.makedirs(output_dir, exist_ok=True)

        evaluator = RotatedCOCOEvaluator(dataset_val, cfg, False, output_dir=output_dir)
        loader = build_detection_test_loader(cfg, dataset_val, mapper=mapper)
        summary_val = inference_on_dataset(trainer.model, loader, evaluator)

        res.update({dataset_val: summary_val})

    res_file = res_dir + f"{model_name}_coco_metrics_again.json"
    with open(res_file, 'w') as f:
        json.dump(res, f)

    return res

def eval_models_in_list(model_weights_files_file, gpu_idx=0):
    prepare_datasets()

    with open(model_weights_files_file, 'rb') as f:
        model_weights_files = pickle.load(f)

    for (i, model_weights_file) in enumerate(model_weights_files):
        print(i, '      ', model_weights_file)
        sys.stdout.flush()
        tag = model_weights_file.split('/')[-3].split('_')[1]
        if len(tag) > 0:
            tag = '_' + tag
        dataset_train = f"{EXPERIMENT_NAME}_train{tag}"
        dataset_val = f"{EXPERIMENT_NAME}_val{tag}"
        cfg_file = '/'.join(model_weights_file.split('/')[:-1]) + '/config.yaml'
        eval_model(model_weights_file, cfg_file, dataset_train, dataset_val, gpu_idx=gpu_idx)


if __name__ == "__main__":
    EXPERIMENT_NAME = 'nanorods'

    parser = argparse.ArgumentParser(description="train maskRCNN")
    parser.add_argument('--model_weights_files_0', type=str, default=None,
                        help='a pkl file containing list of model_weights_files to evaluate on gpu 0')
    parser.add_argument('--model_weights_files_1', type=str, default=None,
                        help='a pkl file containing list of model_weights_files to evaluate on gpu 1')
    args = parser.parse_args()
    model_weights_files_list = [args.model_weights_files_0, args.model_weights_files_1]

#     eval_models_in_list(args.model_weights_files_0, gpu_idx=0)
    threads = []
    for gpu_idx in np.arange(2):
        threads.append(threading.Thread(target=eval_models_in_list, args=(model_weights_files_list[gpu_idx], gpu_idx)))
        threads[-1].start()
        print(f'threds {gpu_idx} started')












