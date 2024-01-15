# Copyright (c) Facebook, Inc. and its affiliates.

"""
Modified from https://github.com/facebookresearch/votenet
Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import sys
import cv2
from .build import DATASETS

import numpy as np
import torch
# import utils.pc_util as pc_util
from torch.utils.data import Dataset
# from utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
#                             get_3d_box_batch_np, get_3d_box_batch_tensor)
# from utils.pc_util import scale_points, shift_scale_points

IGNORE_LABEL = -100
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DATASET_ROOT_DIR = "/home/zhiminc/fastscratch/scans/scans_output"  ## Replace with path to dataset
DATASET_METADATA_DIR = "/home/zhiminc/fastscratch/scans/scans_output" ## Replace with path to dataset

DATASET_ROOT_DIR_obj_2D = "/home/zhiminc/scratch1link"


class ScannetDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 18
        self.num_angle_bin = 1
        self.max_num_obj = 64

        self.type2class = {
            "cabinet": 0,
            "bed": 1,
            "chair": 2,
            "sofa": 3,
            "table": 4,
            "door": 5,
            "window": 6,
            "bookshelf": 7,
            "picture": 8,
            "counter": 9,
            "desk": 10,
            "curtain": 11,
            "refrigerator": 12,
            "showercurtrain": 13,
            "toilet": 14,
            "sink": 15,
            "bathtub": 16,
            "garbagebin": 17,
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.nyu40ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        )
        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        }

        # Semantic Segmentation Classes. Not used in 3DETR
        self.num_class_semseg = 20
        self.type2class_semseg = {
            "wall": 0,
            "floor": 1,
            "cabinet": 2,
            "bed": 3,
            "chair": 4,
            "sofa": 5,
            "table": 6,
            "door": 7,
            "window": 8,
            "bookshelf": 9,
            "picture": 10,
            "counter": 11,
            "desk": 12,
            "curtain": 13,
            "refrigerator": 14,
            "showercurtrain": 15,
            "toilet": 16,
            "sink": 17,
            "bathtub": 18,
            "garbagebin": 19,
        }
        self.class2type_semseg = {
            self.type2class_semseg[t]: t for t in self.type2class_semseg
        }
        self.nyu40ids_semseg = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        )
        self.nyu40id2class_semseg = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids_semseg))
        }

    def angle2class(self, angle):
        raise ValueError("ScanNet does not have rotated bounding boxes.")

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        zero_angle = torch.zeros(
            (pred_cls.shape[0], pred_cls.shape[1]),
            dtype=torch.float32,
            device=pred_cls.device,
        )
        return zero_angle

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        zero_angle = np.zeros(pred_cls.shape[0], dtype=np.float32)
        return zero_angle

    def param2obb(
        self,
        center,
        heading_class,
        heading_residual,
        size_class,
        size_residual,
        box_size=None,
    ):
        heading_angle = self.class2angle(heading_class, heading_residual)
        if box_size is None:
            box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
        centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
        new_centers = np.dot(centers, np.transpose(rot_mat))

        dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
        new_x = np.zeros((dx.shape[0], 4))
        new_y = np.zeros((dx.shape[0], 4))

        for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            crnrs = np.zeros((dx.shape[0], 3))
            crnrs[:, 0] = crnr[0] * dx
            crnrs[:, 1] = crnr[1] * dy
            crnrs = np.dot(crnrs, np.transpose(rot_mat))
            new_x[:, i] = crnrs[:, 0]
            new_y[:, i] = crnrs[:, 1]

        new_dx = 2.0 * np.max(new_x, 1)
        new_dy = 2.0 * np.max(new_y, 1)
        new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

        return np.concatenate([new_centers, new_lengths], axis=1)

@DATASETS.register_module()
class Scannet(Dataset):
    def __init__(
        self, config,
        split_set="train",
        root_dir=None,
        meta_data_dir=None,
        num_points=20000,
        use_color=False,
        use_height=False,
        augment=False,
        subset=0,
        use_random_cuboid=True,
        random_cuboid_min_points=30000,

    ):

        split_set = config.subset
        assert split_set in ["train", "val"]
        if root_dir is None:
            root_dir = DATASET_ROOT_DIR

        if meta_data_dir is None:
            meta_data_dir = DATASET_METADATA_DIR

        self.data_path = root_dir
        # all_scan_names = list(
        #     set(
        #         [
        #             os.path.basename(x)[0:12]
        #             for x in os.listdir(self.data_path)
        #             if x.startswith("scene")
        #         ]
        #     )
        # )

        if split_set in ["train", "val", "test"]:
            split_filenames = os.path.join(meta_data_dir, f"scannetv2_{split_set}.txt")
            with open(split_filenames, "r") as f:
                self.scan_names = f.read().splitlines()
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            # self.scan_names = [
            #     sname for sname in self.scan_names if sname in all_scan_names
            # ]
            print(f"kept {len(self.scan_names)} scans out of {num_scans}")
        else:
            raise ValueError(f"Unknown split name {split_set}")

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment
        self.use_random_cuboid = use_random_cuboid
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]

        self.nump = 20000  ### Number of points from the depth scans

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        file_name = self.scan_names[idx]
        frame_name = os.path.split(file_name)[-1][:-4]
        scene_name = os.path.split(file_name)[-2]
        img_path = os.path.join(DATASET_ROOT_DIR, 'color', file_name[:-3] + 'jpg')
        # depth_path = os.path.join(DATASET_ROOT_DIR, 'depth', file_name[:-3] + 'npy')
        depth_path = os.path.join(DATASET_ROOT_DIR, 'depth', file_name[:-3] + 'png')

        pred_mask_path = os.path.join(DATASET_ROOT_DIR, 'tag2text_pred_dino', scene_name, 'mask' + frame_name + '.npy')
        pred_sem_path = os.path.join(DATASET_ROOT_DIR, 'tag2text_pred_dino', scene_name, frame_name + '.npy')
        pose = os.path.join(DATASET_ROOT_DIR, 'pose', scene_name, frame_name + '.txt')
        depth_intrinsic = os.path.join(DATASET_ROOT_DIR, 'intrinsic', scene_name, 'intrinsic_depth.txt')
        caption_feat = os.path.join(DATASET_ROOT_DIR, 'caption', scene_name, frame_name + '_caption.npy')
        feat_2D = os.path.join(DATASET_ROOT_DIR, 'scene_2D', scene_name, frame_name + '.npy')

        obj_feat_2D = os.path.join(DATASET_ROOT_DIR_obj_2D, 'scene_2D', scene_name, 'obj_' + frame_name + '.npy')
        combine_pred_sem_path = os.path.join(DATASET_ROOT_DIR, 'tag2text_pred_dino', scene_name, 'combine' + frame_name + '.npy')
        combine_pred_sem = np.load(combine_pred_sem_path)

        
        
        pred_mask = np.load(pred_mask_path)
        pred_sem = np.load(pred_sem_path)
        caption_feat = np.load(caption_feat)
        feat_2D = np.load(feat_2D)
        obj_feat_2D = np.load(obj_feat_2D)

        pred_mask = cv2.resize(pred_mask, [224, 224], interpolation=cv2.INTER_NEAREST)

        img = cv2.imread(img_path)
        # depth_img = np.load(depth_path)
        depth_img = cv2.imread(depth_path, -1)



        pose = np.loadtxt(pose)
        depth_intrinsic = np.loadtxt(depth_intrinsic)

        img = cv2.resize(img, (224, 224))


        # depth image to point cloud projection
        depth_shift = 1000.0
        x, y = np.meshgrid(np.linspace(0, depth_img.shape[1] - 1, depth_img.shape[1]),
                           np.linspace(0, depth_img.shape[0] - 1, depth_img.shape[0]))
        uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
        uv_depth[:, :, 0] = x
        uv_depth[:, :, 1] = y
        uv_depth[:, :, 2] = depth_img / depth_shift
        uv_depth = np.reshape(uv_depth, [-1, 3])
        uv_depth = uv_depth[np.where(uv_depth[:, 2] != 0), :].squeeze()

        fx = depth_intrinsic[0, 0]
        fy = depth_intrinsic[1, 1]
        cx = depth_intrinsic[0, 2]
        cy = depth_intrinsic[1, 2]
        bx = depth_intrinsic[0, 3]
        by = depth_intrinsic[1, 3]
        n = uv_depth.shape[0]
        points = np.ones((n, 4))
        X = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx + bx
        Y = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy + by
        points[:, 0] = X
        points[:, 1] = Y
        points[:, 2] = uv_depth[:, 2]
        points_world = np.dot(points, np.transpose(pose))
        points_world = points_world[:, :3]


        # img_proj = np.zeros((480, 640))
        # for i in range(uv_depth.shape[0]):
        #     img_proj[int(uv_depth[i, 1]), int(uv_depth[i, 0])] = uv_depth[i, 2]/3
        #
        # img_proj.resize(480, 640, 1)
        # img_proj = np.repeat(img_proj, 3, axis=2)
        #
        # import matplotlib.pyplot as plt
        # plt.imshow(img_proj)
        # plt.show()
        #
        # plt.imshow(img)
        # plt.show()


        paired_imgpose = np.zeros((points_world.shape[0], 2))
        # paired_imgpose[:, 0] = uv_depth[:, 0]
        # paired_imgpose[:, 1] = uv_depth[:, 1]
        
        combine_pred_sem = combine_pred_sem[:, :, 0]
        paired_imgpose[:, 0] = uv_depth[:, 1].astype(int)
        paired_imgpose[:, 1] = uv_depth[:, 0].astype(int)
        idx = 640 * paired_imgpose[:, 0] + paired_imgpose[:, 1]
        idx = idx.astype(int)
        psd = pred_sem[:, :, 0].flatten()[idx]
        psd[psd == 255] = 0

        
        paired_imgpose[:, 0] = np.around(paired_imgpose[:, 0]/480 * 224)
        paired_imgpose[:, 1] = np.around(paired_imgpose[:, 1]/640 * 224)






        if len(points_world) >= self.nump:
            sel_idx = np.random.choice(len(points_world), self.nump, replace=False)
        else:
            sel_idx = np.random.choice(len(points_world), self.nump, replace=True)
        points_world = points_world[sel_idx]
        paired_imgpose = paired_imgpose[sel_idx]

        pred_mask = pred_mask[:, :, 0]
        pred_sem = pred_sem[:, :, 0]
        psd = psd[sel_idx].astype(int)


        # return 'Scannet', 'sample', (points_world)

        return 'Scannet', 'sample', (points_world.astype(np.float32), paired_imgpose, img, pred_mask, pred_sem, caption_feat[0], feat_2D[0], obj_feat_2D[0], psd)
