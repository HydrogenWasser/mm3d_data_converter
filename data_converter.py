# meg_converter.py
import os
from os import path as osp
import mmengine
from pyquaternion import Quaternion
import json
import numpy as np
import open3d as o3d
import math
pcdClass_names = ['pedestrian', 'unkonwn', 'bicycle', 'motor', 'tri-cycle', 'car', 'breadCar', 'bigTrunk', 'midTrunk', 'smallTrunk', 'bigBus', 'midBus', 'halfHangTrunk'] 
pcdClass_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
pcdCategories = dict(zip(pcdClass_names, pcdClass_order))
# imgClass_order = []
# imgCatagores = dict(zip())
def create_custom_dataset_infos(root_path, info_prefix):
    
    train_infos, val_infos = _fill_trainval_infos(root_path)
    metainfo = {
        'categories': pcdCategories,
        'dataset': 'custom_dataset', 
        'info_version': 1.0,
    }
    
    if train_infos is not None:
        data = dict(data_list=train_infos, metainfo=metainfo)
        info_path = osp.join(root_path,
                             '{}_infos_train_lidar-cam.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)

    if val_infos is not None:
        data['data_list'] = val_infos
        info_val_path = osp.join(root_path,
                                 '{}_infos_val_lidar-cam.pkl'.format(info_prefix))
        mmengine.dump(data, info_val_path)

# 根据数据集内容使用
def add_difficulty_to_annos(bbox, occlusion, truncation):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    height = bbox[3] - bbox[1]
    diff = 0
    easy_mask = np.ones((1, ), dtype=bool)
    moderate_mask = np.ones((1, ), dtype=bool)
    hard_mask = np.ones((1, ), dtype=bool)

    if occlusion > max_occlusion[0] or truncation > max_trunc[0]:
        easy_mask = False
    if occlusion > max_occlusion[1] or truncation > max_trunc[1]:
        moderate_mask = False
    if occlusion > max_occlusion[2] or truncation > max_trunc[2]:
        hard_mask = False

        
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)
    
    if is_easy:
        diff = 0
    elif is_moderate:
        diff = 1
    elif is_hard:
        diff = 2
    else:
        diff = -1

    return diff

def _fill_trainval_infos(root_path):

    train_infos = []
    val_infos = []
    use_camera = True

    trainSet = root_path + '/ImageSets/train.txt'
    valSet = root_path + '/ImageSets/val.txt'
    train_dict  , val_dict = set(), set()
    with open(trainSet, 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')
            train_dict.add(ann)
    with open(valSet, 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')
            val_dict.add(ann)

    totalPoints = os.listdir(root_path + '/points')
    for i in range(len(totalPoints)):
        
        file_name = totalPoints[i][:-4]
        # print(file_name)
        lidar_path = root_path + '/shaped_points/' + file_name + '.bin'
        img_path = root_path + '/undistort_images/' + file_name + '.jpg'
        label_path = root_path + '/labels/' + file_name + '.txt'
        
        mmengine.check_file_exist(lidar_path)
        mmengine.check_file_exist(img_path)
        mmengine.check_file_exist(label_path)
        
        time_stamp_list = file_name.split('_')
        time_stamp = int(time_stamp_list[0][-4:]) + int(time_stamp_list[1]) / (10 * len(time_stamp_list[1]))
        # print(time_stamp)
        info = {
            'sample_idx': i,
            'timestamp': time_stamp,
            'lidar_points': dict(),
            'images': dict(),
            'instances': [],
            'cam_instances': dict(),
        }
        
        # lidar_points 相关参数
        info['lidar_points']['lidar_path'] = lidar_path
        info['lidar_points']['num_pts_feats'] = 4
        info['lidar_points']['Tr_velo_cam'] = np.array([
                                                        [0.79807554, 0.60254895, 0.00319398, 0.18529999999999987],
                                                        [ 0.2647308, -0.34586413, -0.90016421, 0.12779,],
                                                        [-0.54128832, 0.71924458, -0.43553896, -0.12140999999999998],
                                                        [0, 0, 0, 1]
                                                    ])
        info['lidar_points']['Tr_imu_to_velo'] = None

        cameras = [
            'cam62',
            # 'cam63',
            # 'cam64',
        ]


        # image 相关参数
        for cam_name in cameras:
            if cam_name not in info['images']:
                info['images'][cam_name] = dict()
                info['cam_instances'][cam_name] = []
            cam_path = root_path + '/undistort_images/' + file_name + '.jpg'
            info['images'][cam_name]['img_path'] = cam_path
            info['images'][cam_name]['height'] = 1080
            info['images'][cam_name]['width'] = 1920
            info['images'][cam_name]['cam2img'] = np.array([
                                                        [1158.52, 0, 964.76, 0],
                                                        [0, 1153.0, 545.86, 0],
                                                        [0, 0, 1.0, 0],
                                                        [0, 0, 0, 1]
                                                    ])

            info['images'][cam_name]['lidar2cam'] = np.array([
                                                        [0.79807554, 0.60254895, 0.00319398, 0.18529999999999987],
                                                        [ 0.2647308, -0.34586413, -0.90016421, 0.12779,],
                                                        [-0.54128832, 0.71924458, -0.43553896, -0.12140999999999998],
                                                        [0, 0, 0, 1]
                                                    ])
            info['images'][cam_name]['lidar2img'] = info['images'][cam_name]['cam2img'] @ info['images'][cam_name]['lidar2cam']
            
            # print(info['images'][cam_name])
        info['images']['R0_rect'] = np.array([
                                            [1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]
                                        ])


        with open(label_path, 'r', encoding='utf-8') as f:
            # i = 0
            for ann in f.readlines():
                ann = ann.strip('\n')
                ann = ann.split()
                if len(ann):
                    # instances
                    info['instances'].append(dict())
                    info['instances'][-1]['bbox'] = [float(ann[4]), float(ann[5]), float(ann[6]), float(ann[7])]
                    info['instances'][-1]['bbox_label'] = pcdCategories[ann[0]]
                    info['instances'][-1]['bbox_3d'] = [float(ann[11]), float(ann[12]), float(ann[13]), float(ann[10]), float(ann[8]), float(ann[9]), float(ann[14])]
                    info['instances'][-1]['bbox_label_3d'] = pcdCategories[ann[0]]
                    info['instances'][-1]['num_lidar_pts'] = None # 如果没有需要使用的地方，可以用None代替
                    info['instances'][-1]['alpha'] = float(ann[3])
                    info['instances'][-1]['occluded'] = int(float(ann[2]))
                    info['instances'][-1]['truncated'] = int(float(ann[1]))
                    info['instances'][-1]['difficulty'] = int(float(ann[2])) if int(float(ann[2])) < 2 else 2
                    info['instances'][-1]['depth'] = None # 如果没有需要使用的地方，可以用None代替
                    info['instances'][-1]['center_2d'] = None # 如果没有需要使用的地方，可以用None代替
                    info['instances'][-1]['group_id'] = None # 如果没有需要使用的地方，可以用None代替
                    info['instances'][-1]['index'] = None # 如果没有需要使用的地方，可以用None代替
                    
                    # cam_instances
                    info['cam_instances']['cam62'].append(dict())
                    info['cam_instances']['cam62'][-1]['bbox'] = [float(ann[4]), float(ann[5]), float(ann[6]), float(ann[7])]
                    info['cam_instances']['cam62'][-1]['bbox_label'] = pcdCategories[ann[0]]
                    info['cam_instances']['cam62'][-1]['bbox_3d'] = [float(ann[11]), float(ann[12]), float(ann[13]), float(ann[10]), float(ann[8]), float(ann[9]), float(ann[14])]
                    info['cam_instances']['cam62'][-1]['bbox_label_3d'] = pcdCategories[ann[0]]
                    info['cam_instances']['cam62'][-1]['bbox_3d_isvalid'] = True
                    info['cam_instances']['cam62'][-1]['velocity'] = None # 如果没有需要使用的地方，可以用None代替
                    info['cam_instances']['cam62'][-1]['center_2d'] = None # 如果没有需要使用的地方，可以用None代替
                    info['cam_instances']['cam62'][-1]['depth'] = None # 如果没有需要使用的地方，可以用None代替
                    
                            # i += 1
        
        if file_name in train_dict:
            train_infos.append(info)
        else:
            val_infos.append(info)
                



    return train_infos, val_infos

if __name__ == '__main__':
    train_infos, val_infos = _fill_trainval_infos('/data/datasets/custom_dataset')
    print(len(train_infos))
    print(len(val_infos))
