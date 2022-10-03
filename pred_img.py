import os
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2
import regex as re
from PIL import Image
import glob
import json
import IPython
from pathlib import Path
import datetime
import pickle
import json

import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    DatasetEvaluators,
    verify_results,
)
# from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation import RotatedCOCOEvaluator,DatasetEvaluators, inference_on_dataset, coco_evaluation,DatasetEvaluator
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
from detectron2.data import transforms as T
import torch

import data_utils
from helpers_coco import images_annotations_info, create_sub_masks, create_sub_mask_annotation, create_category_annotation, create_image_annotation, create_annotation_format, get_coco_json_format
from helpers_coco import CocoDataset
from helpers_coco import process_img_pixel_annotation
from helpers_coco import list_imgs_in_dir, standardize_color, clean_annotation_mask, remove_noise
from helpers_coco import display_ddicts

from detectron2.config import get_cfg
from detectron2 import model_zoo
from helpers_coco import myVisualizer


model_weights = '/usr/workspace/zhong2/Research/FeedOpt/Questek/Results/Tuning_nanorods/8/output_2classes/setting_8/model_final.pth'
cfg_file = '/'.join(model_weights.split('/')[:-1]) + '/config.yaml'

# cfg = get_cfg()
# cfg.INPUT.MIN_SIZE_TRAIN = 2000  # avoid type mismatch
# cfg.merge_from_file(cfg_file)
# cfg.MODEL.WEIGHTS = model_weights
# cfg.MODEL.DEVICE = 'cpu'

cfg = get_cfg()
try:  # load parameters
    cfg.merge_from_file(cfg_file)
except:   # avoid type mismatch error
    cfg.INPUT.MIN_SIZE_TRAIN = 0
    cfg.merge_from_file(cfg_file)
res_dir = cfg.OUTPUT_DIR

cfg.MODEL.WEIGHTS = model_weights  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
# cfg.MODEL.DEVICE = f'cuda:{gpu_idx}'
cfg.MODEL.DEVICE = 'cuda'








#############################################################################
### dataset

EXPERIMENT_NAME = 'nanorods'

data_dir =  '/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/'
dataset_train = f"{EXPERIMENT_NAME}_train"
dataset_val = f"{EXPERIMENT_NAME}_val"
train_annotation_file = data_dir + f"/{EXPERIMENT_NAME}_train_2classes_coco/train.json"
train_img_dir = data_dir + f"/{EXPERIMENT_NAME}_train/"
val_annotation_file = data_dir + f"/{EXPERIMENT_NAME}_val_2classes_coco/val.json"
val_img_dir = data_dir + f"/{EXPERIMENT_NAME}_val/"

DatasetCatalog.clear()  # resets catalog, helps prevent errors from running cells multiple times
register_coco_instances(dataset_train, {},
                        train_annotation_file,
                        train_img_dir)
register_coco_instances(dataset_val, {},
                        val_annotation_file,
                        val_img_dir)

print('Registered Datasets: ', list(DatasetCatalog.data.keys()))



#############################################################################
### predictor
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

class RotatedPredictor(DefaultPredictor):
    def __init__(self, cfg):

        self.cfg = cfg.clone()  # cfg can be modified by model
        trainer = MyTrainer(self.cfg)
        trainer.resume_or_load(resume=False)   # resume:https://detectron2.readthedocs.io/en/latest/modules/engine.html#detectron2.engine.defaults.DefaultTrainer.resume_or_load
        self.model = trainer.model
        self.model.eval()

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
#             height, width = original_image.shape[:2]
#             image = self.transform_gen.get_transform(original_image).apply_image(original_image)
            image, transforms = T.apply_transform_gens([T.Resize((800, 800))], original_image)
            height = 800
            width = 800

            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

predictor = RotatedPredictor(cfg)
# predictor = DefaultPredictor(cfg)










img_file = '/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/nanorods_train/20210517 0006 HAADF.tif'
#
im = cv2.imread(img_file)
outputs_train = predictor(im)
print(outputs_train)
  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
with open('nanorods_train.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(outputs_train["instances"].to("cpu"), f)
#
#
img_file = '/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/nanorods_val/20210517 0012 HAADF.tif'
im = cv2.imread(img_file)
outputs_val = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#
with open('nanorods_val_0012.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(outputs_val["instances"].to("cpu"), f)

img_file = '/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/nanorods_val/20210517 0002 HAADF.tif'
im = cv2.imread(img_file)
outputs_val = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#
with open('nanorods_val_0002.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(outputs_val["instances"].to("cpu"), f)
