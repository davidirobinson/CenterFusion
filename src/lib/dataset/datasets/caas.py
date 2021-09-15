from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..generic_dataset import GenericDataset

import numpy as np
import cv2
import os

class CAAS(GenericDataset):
  default_resolution = [608, 960]
  num_categories = 10
  class_name = [
    'car', 'truck', 'bus', 'trailer',
    'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
    'traffic_cone', 'barrier']
  cat_ids = {i + 1: i + 1 for i in range(num_categories)}
  max_objs = 128
  def __init__(self, opt, split):
    opt.custom_dataset_ann_path = opt.custom_dataset_img_path + "/annotations/test.json"

    # Load an image to check size
    test_img_name = os.listdir(opt.custom_dataset_img_path + "/image/")[0]
    img = cv2.imread(opt.custom_dataset_img_path + "/image/" + test_img_name)
    opt.input_h = img.shape[0]
    opt.input_w = img.shape[1]

    assert (opt.custom_dataset_img_path != '') and \
      (opt.custom_dataset_ann_path != '') and \
      (opt.num_classes != -1) and \
      (opt.input_h != -1) and (opt.input_w != -1), \
      'The following arguments must be specified for custom datasets: ' + \
      'custom_dataset_img_path, custom_dataset_ann_path, num_classes, ' + \
      'input_h, input_w.'

    img_dir = opt.custom_dataset_img_path
    ann_path = opt.custom_dataset_ann_path
    self.class_name = ['' for _ in range(self.num_categories)]
    self.default_resolution = [opt.input_h, opt.input_w]

    self.images = None
    super().__init__(opt, split, ann_path, img_dir)

    self.num_samples = len(self.images)
    print('Loaded CAAS dataset {} samples'.format(self.num_samples))

  def __len__(self):
    return self.num_samples

  def run_eval(self, results, save_dir):
    pass
