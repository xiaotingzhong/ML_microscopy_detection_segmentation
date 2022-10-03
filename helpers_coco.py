import base64
import IPython
import json
import os
import json
import random
import requests
import regex as re
import glob
from collections import defaultdict
from io import BytesIO
from math import trunc
from pathlib import Path
import contextlib
import io
import datetime
import logging
import xml.etree.ElementTree as X
import math
from itertools import repeat
import shutil
from multiprocessing.pool import ThreadPool
import concurrent.futures

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage                                # (pip install Pillow)
from skimage import measure
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
from PIL import Image                                      # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon
import cv2
import matplotlib.pyplot as plt

from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import RotatedCOCOEvaluator,DatasetEvaluators, inference_on_dataset, coco_evaluation,DatasetEvaluator
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger

##################################################################################################
### Notes
# - the hole checking logic in create_sub_mask_annotation is problematic. It may work but not strictly correct
##################################################################################################



##################################################################################################
### My functions
##################################################################################################

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
color_ids = {
    "(0, 0, 0)": 0,# background
    "(255, 0, 0)": 1, # large_vesicle
    "(0, 255, 0)": 2, # small_vesicle
    "(255, 255, 0)": 3, # hole_vesicle
    "(0, 0, 255)": 4, # hole
    "(255, 255, 255)": 5, # vague_large_vesicle
}

id_colors = {
    0: '(0, 0, 0)',
    1: '(255, 0, 0)',
    2: '(0, 255, 0)',
    3: '(255, 255, 0)',
    4: '(0, 0, 255)',
    5: '(255, 255, 255)'
}

class_info = {
    0: {'name': 'background', 'color': '(0, 0, 0)'},
    1: {'name': 'large_vesicle', 'color': '(255, 0, 0)'},
    2: {'name': 'small_vesicle', 'color': '(0, 255, 0)'},
    3: {'name': 'hole_vesicle', 'color': '(255, 255, 0)'},
    4: {'name': 'hole', 'color': '(0, 0, 255)'},
    5: {'name': 'vague_large_vesicle', 'color': '(255, 255, 255)'}
        }

def process_img_pixel_annotation(input_dir, output_dir):
    """
    read annotation map(s and combine them), clean them and write the standard mask to output_dir png file
        assumes image file name 'img_name', its raw mask file name 'img_name ([\d])' (and no more than 9 masks)
    ------------
    inputs
        fname, str, original image file name
        out_dir, str, output directory
    """
    ### create dirs
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    names = list_imgs_in_dir(input_dir)
    for img_name in names.keys():
        ### combine masks
        mask = np.zeros(cv2.imread(input_dir + img_name).shape, dtype=np.uint8)
        for mask_name in names[img_name]:
            mask_tmp = cv2.imread(input_dir + mask_name)
            mask_tmp = cv2.cvtColor(mask_tmp, cv2.COLOR_BGR2RGB)  # cv2 defaul is BRG color order
            mask_tmp = clean_annotation_mask(mask_tmp)

            ### raise warning for overlapping masks
            ### 1) first kind of overlap: blue hole in other features, hole should overide
            was_blue = (mask_tmp[:, :, 0] == 255).astype(int) + (mask_tmp[:, :, 1] == 255).astype(int)*10 + (mask_tmp[:, :, 2] == 255).astype(int)*100
            was_blue = (was_blue == 100)
            new_blue = (mask[:, :, 0] == 255).astype(int) + (mask[:, :, 1] == 255).astype(int)*10 + (mask[:, :, 2] == 255).astype(int)*100
            new_blue = (new_blue == 100)
            either_blue = was_blue + new_blue

            mask += mask_tmp
            mask[either_blue] = [0, 0, 255]

        ### 2) second kind of overlap: same color channel duplicate, adds to 245
        if len(np.unique(mask)) > 2:
            print('OVERLAPPING mask in: ', mask_name)

            if output_dir[-1] == '/':
                error_dir = output_dir[:-1] + '_errors/'
            else:
                error_dir = output_dir + '_errors/'
            if not os.path.isdir(error_dir):
                os.makedirs(error_dir)

            overlap = ((mask != 255) & (mask != 0)).astype(np.uint8)*255
            cv2.imwrite(error_dir + mask_name, cv2.cvtColor(overlap, cv2.COLOR_RGB2BGR))
            mask = standardize_color(mask)

        ### write combined mask file
        idx = img_name.rfind('.')
        mask_name = img_name[:idx] + '_mask' + img_name[idx:]
        print(mask_name)
        cv2.imwrite(output_dir + mask_name, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))


def process_img_pixel_annotation_one_color(input_dir, output_dir, color):
    """
    read annotation map(s and combine them), clean them and write the standard mask to output_dir png file
        assumes image file name 'img_name', its raw mask file name 'img_name ([\d])' (and no more than 9 masks)
    ------------
    inputs
        fname, str, original image file name
        out_dir, str, output directory
        color = [3, ], RGB values of color channel
    """
    ### create dirs
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    ### combine masks
    color_val = np.sum((np.array(color) > 0).astype(int) * np.array([1, 10, 100]))
    names = list_imgs_in_dir(input_dir)
    print (names)
    for img_name in names.keys():
        mask = np.zeros(cv2.imread(input_dir + img_name).shape, dtype=np.uint8)
        for mask_name in names[img_name]:
            mask_tmp = cv2.imread(input_dir + mask_name)
            mask_tmp = cv2.cvtColor(mask_tmp, cv2.COLOR_BGR2RGB)  # cv2 defaul is BRG color order
            mask_tmp = clean_annotation_mask(mask_tmp)

            was_exist = (mask_tmp[:, :, 0] == 255).astype(int) + (mask_tmp[:, :, 1] == 255).astype(int)*10 + (mask_tmp[:, :, 2] == 255).astype(int)*100
            was_exist = (was_exist == color_val)
            new_exist = (mask[:, :, 0] == 255).astype(int) + (mask[:, :, 1] == 255).astype(int)*10 + (mask[:, :, 2] == 255).astype(int)*100
            new_exist = (new_exist == color_val)
            either_exist = was_exist + new_exist

            mask[either_exist] = color

        ### write combined mask file
        idx = img_name.rfind('.')
        mask_name = img_name[:idx] + '_mask' + img_name[idx:]
        print(mask_name)
        cv2.imwrite(output_dir + mask_name, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))


def list_imgs_in_dir(input_dir):
    """
    assumes image file name 'img_name', its raw mask file name 'img_name ([\d])' (and no more than 9 masks)
    ------------
    inputs
        fname, str, original image file name
        out_dir, str, output directory
    """
    names = defaultdict(list)  ### assumes image name as 'image_name.png', masks as 'image_name (1).png'
    pattern_mask = re.compile(" \(\d\)")
    for f in os.listdir(input_dir):
        if pattern_mask.search(f):
            img_name = f.split('.')[0][:-4] + '.' + f.split('.')[1] # assumes no more than 9 masks
            try: # check all masks have original images
                if img_name in os.listdir(input_dir):
                    names[img_name].append(f)
            except:
                print('!!! image for mask' + f + ' doesn not exist !!!')

    pattern_mask = re.compile("-\d")   ### assumes image name as 'image_name.png', masks as 'image_name-2.png'
    for f in os.listdir(input_dir):
        match = pattern_mask.search(f)
        if match:
            idx = f.rfind('.')
            img_name = f[:idx][:-(match.span()[1] - match.span()[0])] + f[idx:] # assumes no more than 9 masks
            if int(f[match.span()[0]+1]) > 1:
                names[img_name].append(f)
    return names

def clean_annotation_mask(mask):
    """
    clean each mask for non-standard color and small noise masks
    """
    ### clear color
    mask_color_clean = standardize_color(mask, thres_pixel=127)

    # ### clear noises: assumes single channel active per mask
    # channel_active = np.sum(mask_color_clean, axis=(0, 1))
    # channel_active = channel_active > np.max(channel_active) * 0.9 # noise can results in non-zero signal in channel
    # channel_active = np.arange(3)[channel_active]
    # mask_channel = mask_color_clean[:, :, channel_active[0]]
    # tmp = remove_noise(mask_channel, thres_noise=16)

    # mask_clean = np.zeros(mask.shape, dtype=np.uint8)
    # for c in channel_active:
    #     mask_clean[:, :, c] = tmp

    ### clear noises: assumes multiple channels active per mask
    mask_bw = cv2.cvtColor(mask_color_clean, cv2.COLOR_RGB2GRAY)
    tmp = remove_noise(mask_bw, thres_noise=16)

    mask_clean = mask_color_clean
    for c in np.arange(3):
        mask_clean[:, :, c][tmp==0] = 0

    return mask_clean

def standardize_color(mask, thres_pixel=127):
    """
    I may not chose the standard color in procreat annotation, this script standardize colors.
    Also, some random dots on mask
    """
    if len(mask.shape) == 2:
        mask[mask<thres_pixel] = 0
        mask[mask>thres_pixel] = 255
    else:
        for i in range(mask.shape[2]):
            mask[:, :, i][mask[:, :, i]<thres_pixel] = 0
            mask[:, :, i][mask[:, :, i]>thres_pixel] = 255
    return mask

def remove_noise(mask, thres_noise=16):
    """
    some really small areas are likely just noise
        ref: https://stackoverflow.com/questions/60033274/how-to-remove-small-object-in-image-with-python
    """
    mask = cv2.UMat(mask) # https://stackoverflow.com/questions/54249728/opencv-typeerror-expected-cvumat-for-argument-src-what-is-this
    cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 16:
            cv2.drawContours(mask, [c], -1, (0,0,0), -1)
    mask = cv2.UMat.get(mask)
    return mask


class AvgScoreEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir=None):
        self._output_dir = output_dir

    def reset(self):
        self.avg_score = 0
    def process(self, inputs, outputs):
        tmp = []
        for output in outputs:
            tmp += output["instances"].scores
        self.avg_score =np.mean(tmp)
    def evaluate(self):
        # save self.count somewhere, or print it, or return it.
        return {"avg_score": self.avg_score}


##################################################################################################
### Register rotated bounding boxes: code from detectron2 source code
##################################################################################################

def load_coco_json_rbbox(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO
    import logging

    logger = logging.getLogger(__name__)

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            bbox_modes = [BoxMode.XYXY_ABS, BoxMode.XYWH_ABS, BoxMode.XYXY_REL, BoxMode.XYWH_REL, BoxMode.XYWHA_ABS]
            obj["bbox_mode"] = bbox_modes[anno['bbox_mode']]
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts

def register_coco_instances_rbbox(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json_rbbox(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


metaclasses_simple = {'100000001': 'ship',
             '100000002': 'aircraft carrier',
             '100000003': 'warcraft',
             '100000004': 'merchant ship',
             '100000005': 'aircraft carrier',
             '100000006': 'aircraft carrier',
             '100000007': 'destroyer',
             '100000008': 'warcraft',
             '100000009': 'destroyer',
             '100000010': 'amphibious',
             '100000011': 'cruiser',
             '100000012': 'aircraft carrier',
             '100000013': 'aircraft carrier',
             '100000014': 'destroyer',
             '100000015': 'amphibious',
             '100000016': 'amphibious',
             '100000017': 'amphibious',
             '100000018': 'merchant ship',
             '100000019': 'merchant ship',
             '100000020': 'merchant ship',
             '100000022': 'Hovercraft',
             '100000024': 'ship',
             '100000025': 'merchant ship',
             '100000026': 'ship',
             '100000027': 'submarine',
             '100000028': 'ship',
             '100000029': 'ship',
             '100000030': 'merchant ship',
             '100000031': 'aircraft carrier',
             '100000032': 'aircraft carrier',
             '100000033': 'aircraft carrier'}

metaclasses_extensive = {'100000001': 'ship',
              '100000002': 'aircraft carrier',
              '100000003': 'warcraft',
              '100000004': 'merchant ship',
              '100000005': 'Nimitz class aircraft carrier',
              '100000006': 'Enterprise class aircraft carrier',
              '100000007': 'Arleigh Burke class destroyers',
              '100000008': 'WhidbeyIsland class landing craft',
              '100000009': 'Perry class frigate',
              '100000010': 'Sanantonio class amphibious transport dock',
              '100000011': 'Ticonderoga class cruiser',
              '100000012': 'Kitty Hawk class aircraft carrier',
              '100000013': 'Admiral Kuznetsov aircraft carrier',
              '100000014': 'Abukuma-class destroyer escort',
              '100000015': 'Austen class amphibious transport dock',
              '100000016': 'Tarawa-class amphibious assault ship',
              '100000017': 'USS Blue Ridge (LCC-19)',
              '100000018': 'Container ship',
              '100000019': 'OXo|--)',
              '100000020': 'Car carrier([]==[])',
              '100000022': 'Hovercraft',
              '100000024': 'yacht',
              '100000025': 'Container ship(_|.--.--|_]=',
              '100000026': 'Cruise ship',
              '100000027': 'submarine',
              '100000028': 'lute',
              '100000029': 'Medical ship',
              '100000030': 'Car carrier(======|',
              '100000031': 'Ford-class aircraft carriers',
              '100000032': 'Midway-class aircraft carrier',
              '100000033': 'Invincible-class aircraft carrier'}



# As of 0.3 the XYWHA_ABS box is not supported in the visualizer, this is fixed in master branch atm (19/11/20)
class myVisualizer(Visualizer):

    def draw_dataset_dict(self, dic):
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYWHA_ABS) for x in annos]

            labels = [x["category_id"] for x in annos]
#             labels = [np.round(x["bbox"][-1]) for x in annos]
            names = self.metadata.get("thing_classes", None)
            if names:
                labels = [names[i] for i in labels]
            labels = [
                "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
                for i, a in zip(labels, annos)
            ]
            self.overlay_instances(labels=labels, boxes=boxes, masks=masks, keypoints=keypts)

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            sem_seg = cv2.imread(dic["sem_seg_file_name"], cv2.IMREAD_GRAYSCALE)
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
        return self.output




##################################################################################################
### Create COCO dataset
###     https://github.com/chrise96/image-to-coco-json-converter
##################################################################################################

def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # Check to see if we have created a sub-mask...
            pixel_str = str(pixel)
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
               # Create a sub-mask (one bit per pixel) and add to the dictionary
                # Note: we add 1 pixel of padding in each direction
                # because the contours module doesn"t handle cases
                # where pixels bleed to the edge of the image
                sub_masks[pixel_str] = Image.new("1", (width+2, height+2))

            # Set the pixel value to 1 (default is 0), accounting for padding
            sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")

    polygons = []
    segmentations = []
    sub_mask_array = np.array(sub_mask)
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Check if the contour is a hole
        # !!!!!!!!!!!!!!! THIS LOGIC IS PROBLEMATIC !!!!!!!!!!!!!!!
        r_mask = np.zeros_like(np.array(sub_mask), dtype='bool')
        r_mask[np.round(contour[:, 1]).astype('int'), np.round(contour[:, 0]).astype('int')] = 1
        r_mask = ndimage.binary_fill_holes(r_mask)
        vals_in_contour = sub_mask_array[r_mask]
        if vals_in_contour.mean() < 0.5:
            # print(vals_in_contour.mean())
            continue

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)

        if(poly.is_empty):
            # Go to next iteration, dont save empty values in list
            continue

        polygons.append(poly)

        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    return polygons, segmentations

def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list

def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images

def create_annotation_format(polygon, segmentation, image_id, category_id, annotation_id):
    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygon.area

    annotation = {
        "segmentation": segmentation,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "bbox_mode": BoxMode.XYWH_ABS,
        "category_id": category_id,
        "id": annotation_id
    }

    return annotation

def get_coco_json_format():
    # Standard COCO format
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format


def images_annotations_info(mask_dir, img_dir, category_ids, category_colors, mask_format='png', input_format='png', multipolygon_ids = []):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []

    for mask_image in glob.glob(mask_dir + "*." + mask_format):
        # The mask image is *.png but the original image is *.jpg.
        # We make a reference to the original file in the COCO JSON file

        tmp = mask_image.split('/')[-1]
        original_file_name = img_dir + tmp[:tmp.rfind('.')-len('_mask')] + '.' + input_format
        # original_file_name = img_dir + mask_image.split('/')[-1].split('.')[0][:-5] + '.' + input_format
        try:
            Path(original_file_name).is_file()
        except:
            print('Original image file ', original_file_name, ' does not exist!!!' )
            return

        # Open the image and (to be sure) we convert it to RGB
        mask_image_open = Image.open(mask_image).convert("RGB")
        w, h = mask_image_open.size

        # "images" info
        image = create_image_annotation(original_file_name, w, h, image_id)
        images.append(image)

        sub_masks = create_sub_masks(mask_image_open, w, h)
        for color, sub_mask in sub_masks.items():
            category_id = category_colors[color]
            if category_ids['background'] == category_id:
                continue
            else:
                # "annotations" info
                polygons, segmentations = create_sub_mask_annotation(sub_mask)

                # Check if we have classes that are a multipolygon
                if category_id in multipolygon_ids:
                    # Combine the polygons to calculate the bounding box and area
                    multi_poly = MultiPolygon(polygons)

                    annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)

                    annotations.append(annotation)
                    annotation_id += 1
                else:
                    for i in range(len(polygons)):
                        # Cleaner to recalculate this variable
                        segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]

                        annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)

                        annotations.append(annotation)
                        annotation_id += 1
        image_id += 1
    return images, annotations, annotation_id



##################################################################################################
### Visualization
###     https://www.kaggle.com/ericdepotter/visualize-coco-annotations/notebook
##################################################################################################
# Load the dataset json
class CocoDataset():
    def __init__(self, annotation_path, image_dir):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.colors = ['purple', 'red', 'green', 'blue', 'gold', 'white', 'orange', 'salmon', 'pink',
                        'orchid', 'slateblue', 'limegreen', 'seagreen', 'darkgreen', 'olive',
                        'teal', 'aquamarine', 'steelblue', 'powderblue', 'dodgerblue', 'navy',
                        'magenta', 'sienna', 'maroon']

        json_file = open(self.annotation_path)
        self.coco = json.load(json_file)
        json_file.close()

        #self.process_info()
        #self.process_licenses()
        self.process_categories()
        self.process_images()
        self.process_segmentations()

    def display_info(self):
        print('Dataset Info:')
        print('=============')
        for key, item in self.info.items():
            print('  {}: {}'.format(key, item))

        requirements = [['description', str],
                        ['url', str],
                        ['version', str],
                        ['year', int],
                        ['contributor', str],
                        ['date_created', str]]
        for req, req_type in requirements:
            if req not in self.info:
                print('ERROR: {} is missing'.format(req))
            elif type(self.info[req]) != req_type:
                print('ERROR: {} should be type {}'.format(req, str(req_type)))
        print('')

    def display_licenses(self):
        print('Licenses:')
        print('=========')

        requirements = [['id', int],
                        ['url', str],
                        ['name', str]]
        for license in self.licenses:
            for key, item in license.items():
                print('  {}: {}'.format(key, item))
            for req, req_type in requirements:
                if req not in license:
                    print('ERROR: {} is missing'.format(req))
                elif type(license[req]) != req_type:
                    print('ERROR: {} should be type {}'.format(req, str(req_type)))
            print('')
        print('')

    def display_categories(self):
        print('Categories:')
        print('=========')
        for sc_key, sc_val in self.super_categories.items():
            print('  super_category: {}'.format(sc_key))
            for cat_id in sc_val:
                print('    id {}: {}'.format(cat_id, self.categories[cat_id]['name']))
            print('')

    def display_image(self, image_id, show_polys=True, show_bbox=True, show_labels=True, show_crowds=True, use_url=False):
        print('Image:')
        print('======')
        if image_id == 'random':
            image_id = random.choice(list(self.images.keys()))

        # Print the image info
        image = self.images[image_id]
        for key, val in image.items():
            print('  {}: {}'.format(key, val))

        # Open the image
        if use_url:
            image_path = image['coco_url']
            response = requests.get(image_path)
            image = PILImage.open(BytesIO(response.content))

        else:
            image_path = os.path.join(self.image_dir, image['file_name'])
            image = PILImage.open(image_path)

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = "data:image/png;base64, " + base64.b64encode(buffered.getvalue()).decode()

        # Calculate the size and adjusted display size
        max_width = 900
        image_width, image_height = image.size
        adjusted_width = min(image_width, max_width)
        adjusted_ratio = adjusted_width / image_width
        adjusted_height = adjusted_ratio * image_height

        # Create list of polygons to be drawn
        polygons = {}
        bbox_polygons = {}
        rle_regions = {}
        poly_colors = {}
        labels = {}
        print('  segmentations ({}):'.format(len(self.segmentations[image_id])))
        for i, segm in enumerate(self.segmentations[image_id]):
            polygons_list = []
            if segm['iscrowd'] != 0:
                # Gotta decode the RLE
                px = 0
                x, y = 0, 0
                rle_list = []
                for j, counts in enumerate(segm['segmentation']['counts']):
                    if j % 2 == 0:
                        # Empty pixels
                        px += counts
                    else:
                        # Need to draw on these pixels, since we are drawing in vector form,
                        # we need to draw horizontal lines on the image
                        x_start = trunc(trunc(px / image_height) * adjusted_ratio)
                        y_start = trunc(px % image_height * adjusted_ratio)
                        px += counts
                        x_end = trunc(trunc(px / image_height) * adjusted_ratio)
                        y_end = trunc(px % image_height * adjusted_ratio)
                        if x_end == x_start:
                            # This is only on one line
                            rle_list.append({'x': x_start, 'y': y_start, 'width': 1 , 'height': (y_end - y_start)})
                        if x_end > x_start:
                            # This spans more than one line
                            # Insert top line first
                            rle_list.append({'x': x_start, 'y': y_start, 'width': 1, 'height': (image_height - y_start)})

                            # Insert middle lines if needed
                            lines_spanned = x_end - x_start + 1 # total number of lines spanned
                            full_lines_to_insert = lines_spanned - 2
                            if full_lines_to_insert > 0:
                                full_lines_to_insert = trunc(full_lines_to_insert * adjusted_ratio)
                                rle_list.append({'x': (x_start + 1), 'y': 0, 'width': full_lines_to_insert, 'height': image_height})

                            # Insert bottom line
                            rle_list.append({'x': x_end, 'y': 0, 'width': 1, 'height': y_end})
                if len(rle_list) > 0:
                    rle_regions[segm['id']] = rle_list
            else:
                # Add the polygon segmentation
                for segmentation_points in segm['segmentation']:
                    segmentation_points = np.multiply(segmentation_points, adjusted_ratio).astype(int)
                    polygons_list.append(str(segmentation_points).lstrip('[').rstrip(']'))

            polygons[segm['id']] = polygons_list

            # if i < len(self.colors):
            #     poly_colors[segm['id']] = self.colors[i]
            # else:
            #     poly_colors[segm['id']] = 'white'
            poly_colors[segm['id']] = self.colors[segm['category_id']]

            bbox = segm['bbox']
            bbox_points = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1],
                           bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0], bbox[1] + bbox[3],
                           bbox[0], bbox[1]]
            bbox_points = np.multiply(bbox_points, adjusted_ratio).astype(int)
            bbox_polygons[segm['id']] = str(bbox_points).lstrip('[').rstrip(']')

            labels[segm['id']] = (self.categories[segm['category_id']]['name'], (bbox_points[0], bbox_points[1] - 4))
            ### display polygon_id
            # labels[segm['id']] = (segm['id'], (bbox_points[0], bbox_points[1] - 4))

            # Print details
            print('    {}:{}:{}'.format(segm['id'], poly_colors[segm['id']], self.categories[segm['category_id']]))

        # Draw segmentation polygons on image
        html = '<div class="container" style="position:relative;">'
        html += '<img src="{}" style="position:relative;top:0px;left:0px;width:{}px;">'.format(img_str, adjusted_width)
        html += '<div class="svgclass"><svg width="{}" height="{}">'.format(adjusted_width, adjusted_height)

        if show_polys:
            for seg_id, points_list in polygons.items():
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                for points in points_list:
                    html += '<polygon points="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0.5" />'.format(points, fill_color, stroke_color)

        if show_crowds:
            for seg_id, rect_list in rle_regions.items():
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                for rect_def in rect_list:
                    x, y = rect_def['x'], rect_def['y']
                    w, h = rect_def['width'], rect_def['height']
                    html += '<rect x="{}" y="{}" width="{}" height="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0.5; stroke-opacity:0.5" />'.format(x, y, w, h, fill_color, stroke_color)

        if show_bbox:
            for seg_id, points in bbox_polygons.items():
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                html += '<polygon points="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0" />'.format(points, fill_color, stroke_color)

        if show_labels:
            for seg_id, label in labels.items():
                color = poly_colors[seg_id]
                html += '<text x="{}" y="{}" style="fill:{}; font-size: 12pt;">{}</text>'.format(label[1][0], label[1][1], color, label[0])

        html += '</svg></div>'
        html += '</div>'
        html += '<style>'
        html += '.svgclass { position:absolute; top:0px; left:0px;}'
        html += '</style>'
        return html

    def process_info(self):
        self.info = self.coco['info']

    def process_licenses(self):
        self.licenses = self.coco['licenses']

    def process_categories(self):
        self.categories = {}
        self.super_categories = {}
        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']

            # Add category to the categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
            else:
                print("ERROR: Skipping duplicate category id: {}".format(category))

            # Add category to super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id} # Create a new set with the category id
            else:
                self.super_categories[super_category] |= {cat_id} # Add category id to the set

    def process_images(self):
        self.images = {}
        for image in self.coco['images']:
            image_id = image['id']
            if image_id in self.images:
                print("ERROR: Skipping duplicate image id: {}".format(image))
            else:
                self.images[image_id] = image

    def process_segmentations(self):
        self.segmentations = {}
        for segmentation in self.coco['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)



##################################################################################################
### AMPIS
##################################################################################################
def display_ddicts(ddict, outpath=None, dataset='', gt=True, img_path=None,
                   suppress_labels=False, summary=True):
    r"""
    Visualize gt annotations overlaid on the image.

    Displays the image in img_path. Overlays the bounding boxes and segmentation masks of each instance in the image.

    Parameters
    ----------

    ddict: list(dict) or None
        for ground truth- data dict containing masks. The format of ddict is described below in notes.

    outpath: str or path-like object, or None
        If None, figure is displayed with plt.show() and not written to disk
        If string/path, this is the location where figure will be saved to

    dataset: str
        name of dataset, included in filename and figure title.
        The dataset should be registered in both the DatasetCatalog and MetadataCatalog
        for proper plotting. (see detectron2 datasetcatalog for more info.)


    gt: bool
        if True, visualizer.draw_dataset_dict() is used for GROUND TRUTH instances
        if False, visualizer.draw_instance_predictions is used for PREDICTED instances

    img_path: str or path-like object
        if None, img_path is read from ddict (ground truth)
        otherwise, it is a string or path to the image file

    suppress_labels: bool
        if True, class names will not be shown on visualizer

    summary: bool
        If True, prints summary of the ddict to terminal


    Returns
    -------
    None

    Notes
    -----
    The ddict should have the following format:

    .. code-block:: text

        'file_name': str or Path object
            path to image corresponding to annotations
        'mask_format': str
            'polygonmask' if segmentation masks are lists of XY coordinates, or
            'bitmask'  if segmentation masks are RLE encoded segmentation masks
        'height': int
            image height in pixels
        'width': int
            image width in pixels
        'annotations': list(dic)
            list of annotations. See the annotation format below.
        'num_instances': int
            equal to len(annotations)- number of instances present in the image


    The dictionary format for the annotation dictionaries is as follows:

    .. code-block:: text

        'category_id': int
                    numeric class label for the instance.
        'bbox_mode': detectron2.structures.BoxMode object
                describes the format of the bounding box coordinates.
                The default is BoxMode.XYXY_ABS.
        'bbox':  list(int)
            4-element list of bbox coordinates
        'segmentation': list
                    list containing:
                      * a list of polygon coordinates (mask format is polygonmasks)
                      * dictionaries  of RLE mask encodings (mask format is bitmasks)


    """
    if img_path is None:
        img_path = ddict['file_name']
    img_path = Path(img_path)

    if suppress_labels:
        if gt:
            ids = [x['category_id'] for x in ddict['annotations']]
        else:
            ids = ddict['instances'].pred_classes
        u = np.unique(ids)
        metadata = {'thing_classes': ['' for x in u]}
    else:
        metadata = MetadataCatalog.get(dataset)

    visualizer = Visualizer(cv2.imread(str(img_path)), metadata=metadata, scale=1)

    if gt:  # TODO automatically detect gt vs pred?
        vis = visualizer.draw_dataset_dict(ddict)
        n = ddict['num_instances']
    else:
        vis = visualizer.draw_instance_predictions(ddict['instances'])
        n = len(ddict['instances'])

    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    ax.imshow(vis.get_image())
    ax.axis('off')
    ax.set_title('{}\n{}'.format(dataset, img_path.name))
    fig.tight_layout()
    if outpath is not None:
#         fig_path = Path(outpath, '{}-n={}_{}.png'.format(dataset, n, img_path.stem))
        fig_path = Path(outpath)
        fig.savefig(fig_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close(fig)

    if summary:
        summary_string = 'ddict info:\n\tpath: {}\n\tnum_instances: {}'.format(img_path, n)
        print(summary_string)
