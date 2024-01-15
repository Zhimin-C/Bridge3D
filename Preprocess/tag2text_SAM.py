import argparse
import os
import warnings
import cv2
import numpy
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           is_distributed)
from PIL import Image

from Tag2Text.models import tag2text
from Tag2Text import inference_tag2text
import torchvision.transforms as TS
from os.path import join

import matplotlib.pyplot as plt

# Grounding DINO
try:
    import groundingdino
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util import get_tokenlizer
    from groundingdino.util.utils import (clean_state_dict,
                                          get_phrases_from_posmap)
    grounding_dino_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
except ImportError:
    groundingdino = None

# mmdet
try:
    import mmdet
    from mmdet.apis import inference_detector, init_detector
except ImportError:
    mmdet = None

import sys

# segment anything
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader, Dataset

sys.path.append('../')
from utils import apply_exif_orientation  # noqa

# GLIP
try:
    import maskrcnn_benchmark

    from mmdet_sam.predictor_glip import GLIPDemo
except ImportError:
    maskrcnn_benchmark = None

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image



def parse_args():
    parser = argparse.ArgumentParser(
        'Detect-Segment-Anything Demo', add_help=True)
    parser.add_argument('det_config', type=str, help='path to det config file')
    parser.add_argument('det_weight', type=str, help='path to det config file')
    parser.add_argument('--only-det', action='store_true')
    parser.add_argument(
        '--sam-type',
        type=str,
        default='vit_h',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='sam type')
    parser.add_argument(
        '--sam-weight',
        type=str,
        default='../models/sam_vit_h_4b8939.pth',
        help='path to checkpoint file')
    parser.add_argument(
        '--box-thr', '-b', type=float, default=0.2, help='box threshold')
    parser.add_argument(
        '--det-device', '-d', default='cuda', help='Device used for inference')
    parser.add_argument(
        '--sam-device', '-s', default='cuda', help='Device used for inference')
    parser.add_argument('--cpu-off-load', '-c', action='store_true')
    parser.add_argument('--num-worker', '-n', type=int, default=2)

    # Detic param
    parser.add_argument('--use-detic-mask', '-u', action='store_true')

    # GroundingDINO param
    parser.add_argument(
        '--text-thr', type=float, default=0.2, help='text threshold')
    parser.add_argument(
        '--apply-original-groudingdino',
        action='store_true',
        help='use original groudingdino label predict')

    # dist param
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


class SimpleDataset(Dataset):

    def __init__(self, img_ids):
        self.img_ids = img_ids

    def __getitem__(self, item):
        return self.img_ids[item]

    def __len__(self):
        return len(self.img_ids)


def __build_grounding_dino_model(args):
    gdino_args = Config.fromfile(args.det_config)
    model = build_model(gdino_args)
    checkpoint = torch.load(args.det_weight, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model


def __build_glip_model(args):
    assert maskrcnn_benchmark is not None
    from maskrcnn_benchmark.config import cfg
    cfg.merge_from_file(args.det_config)
    cfg.merge_from_list(['MODEL.WEIGHT', args.det_weight])
    cfg.merge_from_list(['MODEL.DEVICE', 'cpu'])
    model = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=args.box_thr,
        show_mask_heatmaps=False)
    return model


def __reset_cls_layer_weight(model, weight):
    if type(weight) == str:
        if get_rank() == 0:
            print(f'Resetting cls_layer_weight from file: {weight}')
        zs_weight = torch.tensor(
            np.load(weight),
            dtype=torch.float32).permute(1, 0).contiguous()  # D x C
    else:
        zs_weight = weight
    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros(
            (zs_weight.shape[0], 1))], dim=1)  # D x (C + 1)
    zs_weight = F.normalize(zs_weight, p=2, dim=0)
    zs_weight = zs_weight.to(next(model.parameters()).device)
    num_classes = zs_weight.shape[-1]

    for bbox_head in model.roi_head.bbox_head:
        bbox_head.num_classes = num_classes
        del bbox_head.fc_cls.zs_weight
        bbox_head.fc_cls.zs_weight = zs_weight


def build_detector(args):
    if 'GroundingDINO' in args.det_config:
        detecter = __build_grounding_dino_model(args)
    elif 'glip' in args.det_config:
        detecter = __build_glip_model(args)
    else:
        config = Config.fromfile(args.det_config)
        if 'init_cfg' in config.model.backbone:
            config.model.backbone.init_cfg = None
        if 'detic' in args.det_config and not args.use_detic_mask:
            config.model.roi_head.mask_head = None
        detecter = init_detector(
            config, args.det_weight, device='cpu', cfg_options={})
    return detecter


def create_positive_dict(tokenized, tokens_positive, labels):
    """construct a dictionary such that positive_map[i] = j,
    if token i is mapped to j label"""

    positive_map_label_to_token = {}

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)

            assert beg_pos is not None and end_pos is not None
            positive_map_label_to_token[labels[j]] = []
            for i in range(beg_pos, end_pos + 1):
                positive_map_label_to_token[labels[j]].append(i)

    return positive_map_label_to_token


def convert_grounding_to_od_logits(logits,
                                   num_classes,
                                   positive_map,
                                   score_agg='MEAN'):
    """
    logits: (num_query, max_seq_len)
    """
    assert logits.ndim == 2
    assert positive_map is not None
    scores = torch.zeros(logits.shape[0], num_classes).to(logits.device)
    # 256 -> 80, average for each class
    # score aggregation method
    if score_agg == 'MEAN':  # True
        for label_j in positive_map:
            scores[:, label_j] = logits[:,
                                        torch.LongTensor(positive_map[label_j]
                                                         )].mean(-1)
    else:
        raise NotImplementedError
    return scores


def run_detector(model, image_path, args):
    pred_dict = {}

    if args.cpu_off_load:
        if 'glip' in args.det_config:
            model.model = model.model.to(args.det_device)
            model.device = args.det_device
        else:
            model = model.to(args.det_device)

    if 'GroundingDINO' in args.det_config:
        image_pil = Image.open(image_path).convert('RGB')  # load image
        image_pil = apply_exif_orientation(image_pil)
        image, _ = grounding_dino_transform(image_pil, None)  # 3, h, w

        if get_rank() == 0:
            warnings.warn(f'text prompt is {args.text_prompt}')

        text_prompt = args.text_prompt.lower()
        text_prompt = text_prompt.strip()
        if not text_prompt.endswith('.'):
            text_prompt = text_prompt + '.'


        if not args.apply_original_groudingdino:
            # custom label name
            custom_vocabulary = text_prompt[:-1].split('.')
            label_name = [c.strip() for c in custom_vocabulary]
            tokens_positive = []
            start_i = 0
            separation_tokens = ' . '
            for _index, label in enumerate(label_name):
                end_i = start_i + len(label)
                tokens_positive.append([(start_i, end_i)])
                if _index != len(label_name) - 1:
                    start_i = end_i + len(separation_tokens)
            tokenizer = get_tokenlizer.get_tokenlizer('bert-base-uncased')
            tokenized = tokenizer(
                args.text_prompt, padding='longest', return_tensors='pt')
            positive_map_label_to_token = create_positive_dict(
                tokenized, tokens_positive, list(range(len(label_name))))

        image = image.to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model(image[None], captions=[text_prompt])

        logits = outputs['pred_logits'].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs['pred_boxes'].cpu()[0]  # (nq, 4)

        if not args.apply_original_groudingdino:
            logits = convert_grounding_to_od_logits(
                logits, len(label_name),
                positive_map_label_to_token)  # [N, num_classes]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > args.box_thr
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        if args.apply_original_groudingdino:
            # get phrase
            tokenlizer = model.tokenizer
            tokenized = tokenlizer(text_prompt)
            # build pred
            pred_labels = []
            pred_scores = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > args.text_thr,
                                                      tokenized, tokenlizer)
                pred_labels.append(pred_phrase)
                pred_scores.append(str(logit.max().item())[:4])
        else:
            scores, pred_phrase_idxs = logits_filt.max(1)
            # build pred
            pred_labels = []
            pred_scores = []
            for score, pred_phrase_idx in zip(scores, pred_phrase_idxs):
                pred_labels.append(label_name[pred_phrase_idx])
                pred_scores.append(str(score.item())[:4])

        pred_dict['labels'] = pred_labels
        pred_dict['scores'] = pred_scores

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        pred_dict['boxes'] = boxes_filt
    elif 'glip' in args.det_config:
        image = cv2.imread(image_path)
        # caption
        text_prompt = args.text_prompt
        text_prompt = text_prompt.lower()
        text_prompt = text_prompt.strip()
        if not text_prompt.endswith('.'):
            text_prompt = text_prompt + '.'
        custom_vocabulary = text_prompt[:-1].split('.')
        label_name = [c.strip() for c in custom_vocabulary]

        top_predictions = model.inference(
            image, args.text_prompt, use_other_text=False)
        scores = top_predictions.get_field('scores').tolist()
        labels = top_predictions.get_field('labels').tolist()

        pred_dict['labels'] = [label_name[i - 1] for i in labels]
        pred_dict['scores'] = scores
        pred_dict['boxes'] = top_predictions.bbox
    else:
        result = inference_detector(model, image_path)
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.box_thr]

        pred_dict['boxes'] = pred_instances.bboxes
        pred_dict['scores'] = pred_instances.scores.cpu().numpy().tolist()
        pred_dict['labels'] = [
            model.dataset_meta['classes'][label]
            for label in pred_instances.labels
        ]
        if args.use_detic_mask:
            pred_dict['masks'] = pred_instances.masks.cpu().numpy()

    if args.cpu_off_load:
        if 'glip' in args.det_config:
            model.model = model.model.to('cpu')
            model.device = 'cpu'
        else:
            model = model.to('cpu')
    return model, pred_dict


def fake_collate(x):
    return x


def main():
    if groundingdino is None and maskrcnn_benchmark is None and mmdet is None:
        raise RuntimeError('detection model is not installed,\
                 please install it follow README')

    args = parse_args()
    if args.cpu_off_load is True:
        if 'cpu' in args.det_device and 'cpu ' in args.sam_device:
            raise RuntimeError(
                'args.cpu_off_load is an invalid parameter due to '
                'detection and sam model are on the cpu.')

    only_det = args.only_det
    cpu_off_load = args.cpu_off_load

    # if 'GroundingDINO' in args.det_config or 'glip' in args.det_config \
    #         or 'Detic' in args.det_config:
    #     assert args.text_prompt

    if args.launcher == 'none':
        _distributed = False
    else:
        _distributed = True

    if _distributed and not is_distributed():
        init_dist(args.launcher)

    det_model = build_detector(args)
    if not cpu_off_load:
        if 'glip' in args.det_config:
            det_model.model = det_model.model.to(args.det_device)
            det_model.device = args.det_device
        else:
            det_model = det_model.to(args.det_device)

    if args.use_detic_mask:
        only_det = True

    if not only_det:
        build_sam = sam_model_registry[args.sam_type]
        sam_model = SamPredictor(build_sam(checkpoint=args.sam_weight))
        if not cpu_off_load:
            sam_model.model = sam_model.model.to(args.sam_device)


    tag2text_checkpoint = '/home/zhiminc/CVPR/playground/Tag2Text/tag2text_swin_14m.pth'



    specified_tags='None'
    # load model

    delete_tag_index = []
    for i in range(3012, 3429):
        delete_tag_index.append(i)
    tag2text_model = tag2text.tag2text_caption(pretrained=tag2text_checkpoint,
                                        image_size=384,
                                        vit='swin_b',
                                        delete_tag_index=delete_tag_index)
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    tag2text_model.threshold = 0.64
    tag2text_model.eval()

    device = 'cuda'

    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])

    tag2text_model = tag2text_model.to(device)



    text_prompt_list = ['cabinet', 'bed',  'chair' , 'sofa' ,  'table' ,  'door' ,  'window' ,
                    'bookshelf' ,  'picture',  'counter', 'desk', 'curtain', 'refrigerator' ,
                    'shower curtain', 'toilet', 'sink', 'bathtub', 'other furniture']


    rgb_path = '/media/zhiminc/78E248A3E2486808/output/color'


    scene_names = sorted(os.listdir(rgb_path))
    for scene_name in scene_names:

        color_names = sorted(os.listdir(join(rgb_path, scene_name)),
                             key=lambda a: int(os.path.basename(a).split('.')[0]))
        for color_name in color_names:
            image_path = join(rgb_path, scene_name, color_name)


            img_pred_base_path = join('/media/zhiminc/78E248A3E2486808/output', 'tag2text_pred_dino', scene_name)
            if not os.path.exists(img_pred_base_path):
                os.mkdir(img_pred_base_path)

            sam_base_path = '/media/zhiminc/78E248A3E2486808/output/sam_mask_path'

            sam_path = join(sam_base_path, scene_name, color_name[:-3] + 'npy')

            LEARNING_MAP = [1, 2, 3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]




            image_pil, image = load_image(image_path)

            seg_labels = np.ones([968, 1296, 3]) * 255
            raw_image = image_pil.resize(
                (384, 384))
            raw_image = transform(raw_image).unsqueeze(0).to(device)


            res = inference_tag2text.inference(raw_image, tag2text_model, specified_tags)

            image_name = color_name


            text_prompt = res[0].replace(' |', ',')
            caption = res[2]

            text_prompt = text_prompt.split(", ")
            # using tag2text



            tag2text_prompt = ''
            for text_tag in text_prompt:
                if text_tag in text_prompt_list:
                    tag2text_prompt = tag2text_prompt + text_tag + ' . '



            path_split = image_path.split('/')
            text_base_path = path_split[:-3]
            text_base_path = '/' + os.path.join(*text_base_path) + '/caption/' + scene_name
            if not os.path.exists(text_base_path):
                os.mkdir(text_base_path)


            with open(text_base_path + '/' + image_name[:-4] + '_text.txt', 'w') as f:
                f.write(str(text_prompt))

            with open(text_base_path + '/' + image_name[:-4] + '_text_filter.txt', 'w') as f:
                f.write(str(tag2text_prompt))

            with open(text_base_path + '/' + image_name[:-4] + '_caption.txt', 'w') as f:
                f.write(str(caption))



            args.text_prompt = tag2text_prompt[:-1]
            if tag2text_prompt == '':
                sam_masks = numpy.load(sam_path)
                sam_masks = cv2.resize(sam_masks, [640, 480], interpolation=cv2.INTER_NEAREST)

                pred_matrix = np.ones_like(sam_masks) * 255
                np.save(img_pred_base_path + '/' + 'mask' + color_name[:-3] + 'npy', sam_masks)
                np.save(img_pred_base_path + '/' + color_name[:-3] + 'npy', pred_matrix)
                continue


            print(f"Tags: {tag2text_prompt}")


            det_model, pred_dict = run_detector(det_model, image_path, args)

            image = cv2.imread(image_path)
            if pred_dict['boxes'].shape[0] == 0:

                continue

            if not only_det:

                if cpu_off_load:
                    sam_model.mode = sam_model.model.to(args.sam_device)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                sam_model.set_image(image)

                transformed_boxes = sam_model.transform.apply_boxes_torch(
                    pred_dict['boxes'], image.shape[:2])
                transformed_boxes = transformed_boxes.to(sam_model.model.device)

                masks, _, _ = sam_model.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False)
                pred_dict['masks'] = masks.cpu().numpy()

                if cpu_off_load:
                    sam_model.model = sam_model.model.to('cpu')

            pred_dict['boxes'] = pred_dict['boxes'].cpu().numpy().tolist()

            pred_matrix = np.ones_like(image)
            pred_matrix = pred_matrix * 255



            NAME_MAP = {'cabinet':2, 'bed': 3,  'chair' :4, 'sofa' :5,  'table' :6,  'door' :7,  'window' :8,
                        'bookshelf' : 9,  'picture': 10,  'counter': 11, 'desk':12, 'curtain':13, 'refrigerator' :14,
                        'shower curtain':15, 'toilet':16, 'sink':17, 'bathtub':18, 'otherfurniture':19}

            for i in range(len(pred_dict['boxes'])):
                label = pred_dict['labels'][i]
                score = pred_dict['scores'][i]
                bbox = pred_dict['boxes'][i]
                label_num = NAME_MAP[label]

                if label not in NAME_MAP:
                    warnings.warn(f'not match predicted label of {label}')
                    continue

                if 'masks' in pred_dict:
                    mask = pred_dict['masks'][i][0]
                    pred_matrix[mask] = label_num
                    encode_mask = mask_util.encode(
                        np.array(mask[:, :, np.newaxis], order='F',
                                 dtype='uint8'))[0]
                    encode_mask['counts'] = encode_mask['counts'].decode()


            pred_matrix = cv2.resize(pred_matrix, [640, 480], interpolation=cv2.INTER_NEAREST)
            sam_masks = numpy.load(sam_path)
            max_sam_index = np.unique(sam_masks).max()
            generated_mask = sam_masks
            pred_labels = np.unique(pred_matrix)[:-1]
            for i in range(pred_labels.shape[0]):
                generated_mask[pred_matrix == pred_labels[i]] = max_sam_index + 1 + i

            unique_generated_mask = np.unique(generated_mask)

            generated_mask_save = generated_mask
            for i in range(unique_generated_mask.shape[0]):
                generated_mask_save[generated_mask == unique_generated_mask[i]] = i

            np.save(img_pred_base_path + '/' + 'mask'+ color_name[:-3] + 'npy', generated_mask_save)
            np.save(img_pred_base_path + '/' + color_name[:-3] + 'npy', pred_matrix)




if __name__ == '__main__':
    main()
