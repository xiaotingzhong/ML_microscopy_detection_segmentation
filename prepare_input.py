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
import shutil
from mpi4py import MPI
import time

import matplotlib.pyplot as plt
from pathlib import Path
from detectron2.data.datasets import register_coco_instances
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
)

from helpers_coco import images_annotations_info, create_sub_masks, create_sub_mask_annotation, create_category_annotation, create_image_annotation, create_annotation_format, get_coco_json_format
from helpers_coco import CocoDataset
from helpers_coco import process_img_pixel_annotation, process_img_pixel_annotation_one_color
from helpers_coco import list_imgs_in_dir, standardize_color, clean_annotation_mask, remove_noise

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
import datetime
print("----------------------------------------------------------")
print('Begin at:   ', datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))



### Variables to change
color_targets = [2]
tag = '_aug2'
color_target = comm.scatter(color_targets, root=0)
# input_dir = '/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/vesicles_raw/'
input_dir = f'/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/vesicles_train{tag}_raw/'
mask_dir = f'/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/vesicles_train{tag}_mask_{color_target}/'


### Dataset properties
# Label ids of the dataset
category_ids = {
    "background": 0,
    "large_vesicle": 1,
    "small_vesicle": 2,
    "hole_vesicle": 3,
    "hole": 4,
    "vague_large_vesicle": 5
}

# Define which colors match which categories in the images
category_colors = {
    "(0, 0, 0)": 0,# background
    "(255, 0, 0)": 1, # large_vesicle
    "(0, 255, 0)": 2, # small_vesicle
    "(255, 255, 0)": 3, # hole_vesicle
    "(0, 0, 255)": 4, # hole
    "(255, 255, 255)": 5, # vague_large_vesicle
}

color_categories = dict()
for key in category_colors.keys():
    color_categories[category_colors[key]] = np.array(list(eval(key)))


### Prepare clean masks from hand annotations
if color_target == 'all':
    process_img_pixel_annotation(input_dir, mask_dir)
else:
    process_img_pixel_annotation_one_color(input_dir, mask_dir, color=color_categories[color_target])

mask_val_dir = f'/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/vesicles_val_mask_{color_target}/'
os.makedirs(mask_val_dir, exist_ok=True)
for img_name in os.listdir('/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/vesicles_val'):
    idx = img_name.rfind('.')
    mask_name = img_name[:idx] + '_mask' + img_name[idx:]
    if os.path.isfile(mask_dir + mask_name) and mask_name[-len('DS_Store'):] != 'DS_Store':
        shutil.move(mask_dir + mask_name, mask_val_dir + mask_name)


### Prepare coco dataset
# for keyword in ['train', 'val']:
for keyword in ['train']:

    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()

    # Create category section
    categories = create_category_annotation(category_ids)
    coco_format["categories"] = categories[1:]  # having 0: backgournd superclass will mess up the training in detectron

    # Create images and annotations sections
    input_dir = f'/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/vesicles_{keyword}{tag}/'
    mask_dir = f'/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/vesicles_{keyword}{tag}_mask_{color_target}/'
    coco_dir = f'/usr/workspace/zhong2/Research/FeedOpt/Questek/Data/vesicles_{keyword}{tag}_coco_{color_target}/'
    if not os.path.isdir(coco_dir):
        os.makedirs(coco_dir)

    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_dir, input_dir, category_ids, category_colors)

    with open(coco_dir + "/{}.json".format(keyword),"w") as outfile:
        json.dump(coco_format, outfile)

    print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_dir))


time.sleep(1)
print('End at:   ', datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
print("----------------------------------------------------------")
