# Offline Preprocessing


ScanNet Preprocess: We utilize the same code as in [DepthContrast](https://github.com/facebookresearch/DepthContrast/tree/main/data/scannet)

Download pre-trained models in ../models
```
wget -P ../models/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/tag2text_swin_14m.pth
```

Generate SAM Masks
```
python sam_mask.py 
```

Generate Grounding SAM Results with Tag2Text
```
python tag2text_SAM.py configs/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth 
```
Extract 2D features
```
python Scene_2D_feats.py
```
```
python pred_seg.py
```



