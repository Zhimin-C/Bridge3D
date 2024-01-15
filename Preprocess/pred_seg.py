import torchvision.transforms as T
import configparser
import yaml
from datetime import datetime as dt
import os
from os.path import join
import cv2
import numpy as np
import imageio
_MEAN_PIXEL_IMAGENET = [0.485, 0.456, 0.406]
_STD_PIXEL_IMAGENET = [0.229, 0.224, 0.225]
from maskclip_model import maskClipFeatureExtractor
import torch
import matplotlib.pyplot as plt

def generate_config(file):
    with open(file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        config["datetime"] = dt.today().strftime("%d%m%y-%H%M")

    return config

class Preprocessing:
    """
    Use the ImageNet preprocessing.
    """

    def __init__(self):
        normalize = T.Normalize(mean=_MEAN_PIXEL_IMAGENET, std=_STD_PIXEL_IMAGENET)
        self.preprocessing_img = normalize

    def __call__(self, image):
        return self.preprocessing_img(image)
config = {}
config['text_embeddings_path'] = '/home/zhiminc/CVPR/CLIP2Scene/scannet_ViT16_clip_text.pth'
config['visual_projs_path'] = '/home/zhiminc/CVPR/CLIP2Scene/ViT16_clip_weights.pth'
config['text_categories'] = 20
config['maskclip_checkpoint'] = '/home/zhiminc/CVPR/CLIP2Scene/ViT16_clip_backbone.pth'
model_images = maskClipFeatureExtractor(config, preprocessing=Preprocessing())

rgb_path = '/media/zhiminc/78E248A3E2486808/output/color'
scene_names = sorted(os.listdir(rgb_path))
for scene_name in scene_names:
    color_names = sorted(os.listdir(join(rgb_path, scene_name)),
                         key=lambda a: int(os.path.basename(a).split('.')[0]))
    for color_name in color_names:
        image_path = join(rgb_path, scene_name, color_name)
        img = imageio.imread(image_path) / 255
        img = cv2.resize(img, [640, 480])
        img = torch.from_numpy(img.transpose((2, 0, 1)).astype(np.float32))
        img = img.unsqueeze(0)
        maskcip_pred = model_images(img)[1]

        img_pred_base_path = join('/media/zhiminc/78E248A3E2486808/output', 'tag2text_pred_dino', scene_name)
        image_name = color_name
        samdino_seg = np.load(img_pred_base_path + '/' + image_name[:-3] + 'npy')
        samdino_seg = torch.from_numpy(samdino_seg[:, :, 0])
        samdino_seg = samdino_seg.unsqueeze(0)
        for index in np.unique(samdino_seg)[:-1]:
            maskcip_pred[samdino_seg==index] = index

        aa = np.load(img_pred_base_path + '/' + image_name[:-3] + 'npy')
        pred_matrix_color = np.zeros_like(aa)
        maskcip_pred = maskcip_pred.permute(1, 2 , 0)
        maskcip_pred = maskcip_pred.repeat(1, 1, 3)

        pred_matrix_color = np.zeros_like(aa)

        for i in np.unique(maskcip_pred):
            pred_matrix_color[maskcip_pred[:, :, 0] == i] = np.concatenate([np.random.random(3)]) * 255

        plt.imsave(img_pred_base_path + '/' + 'combine' + color_name, (pred_matrix_color / 255).astype(float))
        np.save(img_pred_base_path + '/' + 'combine'+ color_name[:-3] + 'npy', maskcip_pred)

