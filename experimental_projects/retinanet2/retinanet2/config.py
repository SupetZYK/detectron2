# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN

def add_retinanet2_config(cfg):
    """
    Add config for Retinanet2.
    """
    # ---------------------------------------------------------------------------- #
    # TorchResFPN options， add new backbone
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.TORCH_RES_FPN = CN()
    # depth for torch resnet
    cfg.MODEL.TORCH_RES_FPN.DEPTH = 50
    # fpn feat size
    cfg.MODEL.TORCH_RES_FPN.FPN_FS = 256
    # whether to pretrain
    cfg.MODEL.TORCH_RES_FPN.PRETRAIN = True

    # ---------------------------------------------------------------------------- #
    # RetinaNet2 Head
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.RETINANET2 = CN()

    # This is the number of foreground classes.
    cfg.MODEL.RETINANET2.NUM_CLASSES = 80
    cfg.MODEL.RETINANET2.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.RETINANET2.FPN_FEAT_SIZE = 256

    # anchor config
    cfg.MODEL.RETINANET2.ANCHOR_SIZES = [32, 64, 128, 256, 512]
    cfg.MODEL.RETINANET2.ANCHOR_RATIOS = [0.5, 1., 2.]
    cfg.MODEL.RETINANET2.ANCHOR_SCALES = [1., pow(2, 1./3), pow(2, 2./3)]
    cfg.MODEL.RETINANET2.ANCHOR_OFFSET = 0.

    # loss related
    # allow low quality match
    cfg.MODEL.RETINANET2.LOW_QUALITY_MATCH = True

    # test config
    cfg.MODEL.RETINANET2.TEST_SCORE_THRESH = 0.05
    cfg.MODEL.RETINANET2.NMS_THRESH_TEST = 0.5
    cfg.MODEL.RETINANET2.BBOX_REG_WEIGHT = [1, 1, 1, 1]
    # input
    cfg.MODEL.RETINANET2.PIXEL_MEAN = [103.530, 116.280, 123.675]
    cfg.MODEL.RETINANET2.PIXEL_STD = [57.375, 57.120, 58.395]

    