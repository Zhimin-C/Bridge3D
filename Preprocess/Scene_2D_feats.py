import clip
from PIL import Image
import torch
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from PIL import Image


rgb_path = '/media/zhiminc/78E248A3E2486808/output/color'
caption_path = '/media/zhiminc/78E248A3E2486808/output/caption'
img_pool_base_path = '/media/zhiminc/78E248A3E2486808/output/scene_2D'
if not os.path.exists(img_pool_base_path):
    os.mkdir(img_pool_base_path)

scene_names = sorted(os.listdir(rgb_path))
for scene_name in scene_names:
    color_names = sorted(os.listdir(os.path.join(rgb_path, scene_name)),
                         key=lambda a: int(os.path.basename(a).split('.')[0]))
    for color_name in color_names:

        img_pool_path = os.path.join(img_pool_base_path, scene_name)
        if not os.path.exists(img_pool_path):
            os.mkdir(img_pool_path)


        if int(color_name[:-4])%5 != 0:
            continue
        image_path = os.path.join(rgb_path, scene_name, color_name)
        image = Image.open(image_path)
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        model = AutoModel.from_pretrained('facebook/dinov2-large')
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            pooler_output = outputs.pooler_output

        last_hidden_states = last_hidden_states[:, 1:, :].contiguous().transpose(1, 2).contiguous().view(1, 1024, 16, 16)

        np.save(  os.path.join(img_pool_path, color_name[:-4] + '.npy'), np.array(pooler_output.cpu()))

        np.save(os.path.join(img_pool_path, 'obj_' + color_name[:-4] + '.npy'), np.array(last_hidden_states.cpu()))





