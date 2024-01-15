import json

f = open('/home/zhiminc/Downloads/scannet_train.coco.json')

# f = open('/home/zhiminc/Downloads/scannet_train.coco_pred.json')
data = json.load(f)


for i in data['annotations']:
    print(i)

print(1)
