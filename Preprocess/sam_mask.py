import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from os.path import join

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "/home/zhiminc/CVPR/playground/models/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    # Foggy driving (zero-shot evaluate) is more challenging than other dataset, so we use a larger points_per_side
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

# mask_generator = SamAutomaticMaskGenerator(sam)

# mask_generator = SamAutomaticMaskGenerator(sam)
base_path = '/media/zhiminc/78E248A3E2486808/output'
rgb_path = '/media/zhiminc/78E248A3E2486808/output/color'

scene_names = sorted(os.listdir(rgb_path))

for scene in scene_names:
    color_names = sorted(os.listdir(os.path.join(rgb_path, scene)))

    skip_num = 5
    counter = skip_num
    for frame in range(len(color_names)):
        image_path = join(rgb_path, scene, str(frame) +'.jpg')
        counter += 1
        if counter < skip_num:
            continue
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 480))
        counter = 0

        sam_mask_path = join(base_path, 'sam_mask_path', scene)
        if not os.path.exists(sam_mask_path):
            os.mkdir(sam_mask_path)
        if os.path.exists(sam_mask_path + '/' + str(frame) + '.npy'):
            continue




        masks = mask_generator.generate(image)
        print(len(masks))
        print(masks[0].keys())


        output_masks = np.zeros_like(image)
        for i in range(len(masks)):
            output_masks[masks[i]['segmentation']] = np.ones(3) * (i + 1)

        np.save(sam_mask_path + '/' + str(frame) + '.npy', output_masks)
        #
        # pred_matrix_color = np.zeros_like(output_masks)
        # for i in np.unique(output_masks):
        #     pred_matrix_color[output_masks[:, :, 0] == i] = np.concatenate([np.random.random(3)]) * 255
        # plt.imshow(pred_matrix_color)
        # plt.show()