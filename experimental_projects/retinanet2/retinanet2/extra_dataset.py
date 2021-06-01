from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

_root = '/data/dataset'
register_coco_instances('coco_cls_3_train', {}, _root + '/coco/annotations/coco_cls_3_train.json', _root + '/coco/train2017')
register_coco_instances('coco_cls_3_val', {}, _root + '/coco/annotations/coco_cls_3_val.json', _root + '/coco/val2017')

MetadataCatalog.get('coco_cls_3_train').thing_classes = ['person', 'car', 'bus']
MetadataCatalog.get('coco_cls_3_val').thing_classes = ['person', 'car', 'bus']
