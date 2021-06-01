from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from experimental_projects.retinanet2 import retinanet2

import cv2
import random

def main():
    ds_name = 'coco_cls_3_val'
    metadata = MetadataCatalog.get(ds_name)
    data_dicts = DatasetCatalog.get(ds_name)

    for idx, d in enumerate(random.sample(data_dicts, 3)):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(f'test-{idx}.jpg', vis.get_image()[:, :, ::-1])

if __name__ == "__main__":
    main()