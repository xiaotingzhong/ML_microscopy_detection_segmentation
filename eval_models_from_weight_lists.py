"""
This script takes in two files containing lists of model weight files,
give these weights to models and evaluate models using two GPUs.
All models are basically default COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
    with different number of classes. Solver is different in each model but solver does not affect evaluation
"""
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

from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)
from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader


def prepare_datasets(EXPERIMENT_NAME='vesicles'):
    DatasetCatalog.clear()  # resets catalog, helps prevent errors from running cells multiple times
    for color in ['1', '2', '3', 'all']:
        dataset_train = f"{EXPERIMENT_NAME}_train_{color}"
        dataset_val = f"{EXPERIMENT_NAME}_val_{color}"
        data_dir = '/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/'
        train_annotation_file = data_dir + f"/{EXPERIMENT_NAME}_train_coco_{color}/train.json"
        train_img_dir = data_dir + f"/{EXPERIMENT_NAME}_train/"
        val_annotation_file = data_dir + f"/{EXPERIMENT_NAME}_val_coco_{color}/val.json"
        val_img_dir = data_dir + f"/{EXPERIMENT_NAME}_val/"

        register_coco_instances(dataset_train, {},
                                train_annotation_file,
                                train_img_dir)
        register_coco_instances(dataset_val, {},
                                val_annotation_file,
                                val_img_dir)
    print('Registered Datasets: ', list(DatasetCatalog.data.keys()))
    return

def eval_model(model_weights_file, dataset_train, dataset_val, gpu_idx=0, cfg_file=None):
    ### Prepare predictor model
    cfg = get_cfg()
    if cfg_file:
        try:  # load parameters
            cfg.merge_from_file(cfg_file)
        except:   # avoid type mismatch error
            cfg.INPUT.MIN_SIZE_TRAIN = 0
            cfg.merge_from_file(cfg_file)
        res_dir = cfg.output_dir
    else:
        cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
        cfg.INPUT.MASK_FORMAT = 'polygon'
        cfg.INPUT.FORMAT = 'RGB'
        n_classes = eval(model_weights_file.split('/')[-2].split('_')[1])
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
        res_dir = '/'.join(model_weights_file.split('/')[:-1]) + '/'
    model_name = model_weights_file.split('/')[-1].split('.')[0]

    cfg.MODEL.WEIGHTS = model_weights_file  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    cfg.MODEL.DEVICE = f'cuda:{gpu_idx}'

    ### Evaluate with predictor
    res = {}
    predictor = DefaultPredictor(cfg)
    if dataset_train:
        output_dir = res_dir+f"{model_name}_output_train"
        os.makedirs(output_dir, exist_ok=True)

        evaluator = COCOEvaluator(dataset_train, output_dir=output_dir)
        loader = build_detection_test_loader(cfg, dataset_train)
        summary_train = inference_on_dataset(predictor.model, loader, evaluator)

        res.update({dataset_train: summary_train})

    if dataset_val:
        output_dir = res_dir+f"{model_name}_output_val"
        os.makedirs(output_dir, exist_ok=True)

        evaluator = COCOEvaluator(dataset_val, output_dir=output_dir)
        loader = build_detection_test_loader(cfg, dataset_val)
        summary_val = inference_on_dataset(predictor.model, loader, evaluator)

        res.update({dataset_val: summary_val})

    res_file = res_dir + f"{model_name}_coco_metrics.json"
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
        color = model_weights_file.split('/')[-3].split('_')[1]
        dataset_train = f"{EXPERIMENT_NAME}_train_{color}"
        dataset_val = f"{EXPERIMENT_NAME}_val_{color}"
        eval_model(model_weights_file, dataset_train, dataset_val, gpu_idx=gpu_idx)


if __name__ == "__main__":
    EXPERIMENT_NAME = 'vesicles'

    parser = argparse.ArgumentParser(description="train maskRCNN")
    parser.add_argument('--model_weights_files_0', type=str, default=None,
                        help='a pkl file containing list of model_weights_files to evaluate on gpu 0')
    parser.add_argument('--model_weights_files_1', type=str, default=None,
                        help='a pkl file containing list of model_weights_files to evaluate on gpu 1')
    args = parser.parse_args()
    model_weights_files_list = [args.model_weights_files_0, args.model_weights_files_1]

    if not args.model_weights_files_1:  # only one file to evaluate
        eval_models_in_list(args.model_weights_files_0, gpu_idx=0)
    else:
        threads = []
        for gpu_idx in np.arange(2):
            threads.append(threading.Thread(target=eval_models_in_list, args=(model_weights_files_list[gpu_idx], gpu_idx)))
            threads[-1].start()
            print(f'threds {gpu_idx} started')





