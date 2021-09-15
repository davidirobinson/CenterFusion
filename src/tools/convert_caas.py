# Copyright (c) Xingyi Zhou. All Rights Reserved
'''
caas rosbag pre-processing script.
This file convert the caas rosbag annotation into COCO format.
'''
import os
import csv
import json
import numpy as np
import cv2
from natsort import natsorted
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

import _init_paths
from utils.pointcloud import RadarPointCloudWithVelocity as RadarPointCloud
from nuScenes_lib.utils_radar import map_pointcloud_to_image


# DATA_PATH = '/media/drobinson/2tbexternal/caas/calibration/data/centerfusion_format/'
DATA_PATH = '/media/drobinson/2tbexternal/caas/2021-07-17_extracted_79de6666-b484-44b1-8202-ea1827abf727/centerfusion_format/'
OUT_PATH = DATA_PATH + '/annotations/'

DEBUG = False
CATS = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
SENSOR_ID = {'RADAR_FRONT': 7, 'RADAR_FRONT_LEFT': 9,
  'RADAR_FRONT_RIGHT': 10, 'RADAR_BACK_LEFT': 11,
  'RADAR_BACK_RIGHT': 12,  'LIDAR_TOP': 8,
  'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2,
  'CAM_BACK_RIGHT': 3, 'CAM_BACK': 4, 'CAM_BACK_LEFT': 5,
  'CAM_FRONT_LEFT': 6}

USED_SENSOR = ['CAM_FRONT', 'CAM_FRONT_RIGHT',
  'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
  'CAM_FRONT_LEFT']

RADARS_FOR_CAMERA = {
  'CAM_FRONT_LEFT':  ["RADAR_FRONT_LEFT", "RADAR_FRONT"],
  'CAM_FRONT_RIGHT': ["RADAR_FRONT_RIGHT", "RADAR_FRONT"],
  'CAM_FRONT':       ["RADAR_FRONT_RIGHT", "RADAR_FRONT_LEFT", "RADAR_FRONT"],
  'CAM_BACK_LEFT':   ["RADAR_BACK_LEFT", "RADAR_FRONT_LEFT"],
  'CAM_BACK_RIGHT':  ["RADAR_BACK_RIGHT", "RADAR_FRONT_RIGHT"],
  'CAM_BACK':        ["RADAR_BACK_RIGHT","RADAR_BACK_LEFT"]}

NUM_SWEEPS = 1
suffix1 = '_{}sweeps'.format(NUM_SWEEPS) if NUM_SWEEPS > 1 else ''
OUT_PATH = OUT_PATH + suffix1 + '/'

CAT_IDS = {v: i + 1 for i, v in enumerate(CATS)}

def _rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha

def _bbox_inside(box1, box2):
  return box1[0] > box2[0] and box1[0] + box1[2] < box2[0] + box2[2] and \
         box1[1] > box2[1] and box1[1] + box1[3] < box2[1] + box2[3]

ATTRIBUTE_TO_ID = {
  '': 0, 'cycle.with_rider' : 1, 'cycle.without_rider' : 2,
  'pedestrian.moving': 3, 'pedestrian.standing': 4,
  'pedestrian.sitting_lying_down': 5,
  'vehicle.moving': 6, 'vehicle.parked': 7,
  'vehicle.stopped': 8}

def main():
  if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)

  split = "test"
  data_path = DATA_PATH
  out_path = OUT_PATH + '{}.json'.format(split)
  categories_info = [{'name': CATS[i], 'id': i + 1} for i in range(len(CATS))]
  ret = {'images': [], 'annotations': [], 'categories': categories_info,
          'videos': [], 'attributes': ATTRIBUTE_TO_ID, 'pointclouds': []}
  num_images = 0
  num_anns = 0
  num_videos = 0

  image_files = natsorted(os.listdir(data_path + "/image/"))
  radar_files = natsorted(os.listdir(data_path + "/radar/"))
  assert len(image_files) == len(radar_files)

  for image_file, radar_file in zip(image_files, radar_files):
    # Load image
    image_data = "/image/" + image_file
    img = cv2.imread(data_path + image_data)
    width, height = img.shape[1], img.shape[0]
    num_images += 1

    # Treat car as origin
    global_from_car_translation = np.array([0,0,0])
    global_from_car_rotation = np.array([0,0,0,0])

    # TODO(drobinson): Should we encoode the offset of the sensor here?
    #                  This might be important for pillar height!
    # (Pdb) cs_record['translation']
    # [1.72200568478, 0.00475453292289, 1.49491291905]
    # (Pdb) cs_record['rotation']
    # [0.5077241387638071, -0.4973392230703816, 0.49837167536166627, -0.4964832014373754]
    car_from_sensor_translation = np.array([0,0,0])
    car_from_sensor_rotation = np.array([0,0,0,0])

    global_from_car = transform_matrix(global_from_car_translation,
      Quaternion(global_from_car_rotation), inverse=False)
    car_from_sensor = transform_matrix(
      car_from_sensor_translation, Quaternion(car_from_sensor_rotation),
      inverse=False)
    global_from_sensor = np.dot(global_from_car, car_from_sensor)

    # rotation only!
    vel_global_from_car = transform_matrix(np.array([0,0,0]),
      Quaternion(global_from_car_rotation), inverse=False)
    vel_car_from_sensor = transform_matrix(np.array([0,0,0]),
      Quaternion(car_from_sensor_rotation), inverse=False)
    velocity_global_from_sensor = np.dot(vel_global_from_car, vel_car_from_sensor)

    # Hardcode monocam intrinsics for now
    # NOTE: We're using undistort images for this demo
    camera_intrinsic = np.array(
      [[419.8938293457031, 0.0, 474.5676574707031],
       [0.0, 419.7181396484375, 324.60125732421875],
       [0.0, 0.0, 1.0]])

    calib = np.eye(4, dtype=np.float32)
    calib[:3, :3] = camera_intrinsic
    calib = calib[:3]

    #
    # Get radar pointclouds
    # incoming format: x,y,z,range,ranazimuthge,elevation,doppler,magnitude,snr,rcs
    # outgoing format: x,y,z,dyn_prop,id,rcs,vx,vy,vx_comp,vy_comp,is_quality_valid,ambig_state,x_rms,y_rms,invalid_state,pdh0,vx_rms,vy_rms
    # ref: CenterFusion/src/tools/nuscenes-devkit/python-sdk/nuscenes/utils/data_classes.py
    #
    all_radar_pcs = np.zeros((18, 0))

    with open(data_path + "/radar/" + radar_file, newline='') as csvfile:
      radar_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      for row in radar_reader:
        if row[0] == 'x':
          continue # skip header

        radar_pcs = np.zeros((18, 1))
        radar_pcs[0] = float(row[0]) # x
        radar_pcs[1] = float(row[1]) # y
        radar_pcs[2] = float(row[2]) # z
        radar_pcs[8] = 0 # vx
        radar_pcs[9] = float(row[6]) # doppler -> vz (or vy more likley?)
        radar_pcs[10] = 1 # valid

        all_radar_pcs = np.hstack((all_radar_pcs, radar_pcs))

    # image information in COCO format
    image_info = {'id': num_images,
                  'file_name': image_data,
                  'calib': calib.tolist(),
                  'video_id': num_videos,
                  'frame_id': 0, # Only one cam/radar sensor pair
                  'sensor_id': 0, # Only one cam/radar sensor pair
                  'sample_token': "no sample token!",
                  'trans_matrix': global_from_sensor.tolist(),
                  'velocity_trans_matrix': velocity_global_from_sensor.tolist(),
                  'width': width,
                  'height': height,
                  'pose_record_trans': global_from_car_translation.tolist(),
                  'pose_record_rot': global_from_car_rotation.tolist(),
                  'cs_record_trans': car_from_sensor_translation.tolist(),
                  'cs_record_rot': car_from_sensor_rotation.tolist(),
                  'radar_pc': all_radar_pcs.tolist(),
                  'camera_intrinsic': camera_intrinsic.tolist()}
    ret['images'].append(image_info)

    if DEBUG:
      img_path = data_path + image_info['file_name']
      img = cv2.imread(img_path)
      # plot radar point clouds
      pc = np.array(image_info['radar_pc'])
      cam_intrinsic = np.array(image_info['calib'])[:,:3]
      points, coloring, _ = map_pointcloud_to_image(pc, cam_intrinsic)
      for i, p in enumerate(points.T):
        img = cv2.circle(img, (int(p[0]), int(p[1])), 5, (255,0,0), -1)

      cv2.imshow('img', img)
      cv2.waitKey()

  print('{} {} images {} boxes'.format(
    split, len(ret['images']), len(ret['annotations'])))
  print('out_path', out_path)
  json.dump(ret, open(out_path, 'w'))


if __name__ == '__main__':
  main()
