# Offline Preprocessing


ScanNet Preprocess: We utilize the same code as in [DepthContrast](https://github.com/facebookresearch/DepthContrast/tree/main/data/scannet)

Generate SAM Masks
```
python sam_mask.py
```

Generate Grounding SAM Results with Tag2Text
```
python tag2text_SAM.py
```
Extract 2D features
```
python Scene_2D_feats.py
```

