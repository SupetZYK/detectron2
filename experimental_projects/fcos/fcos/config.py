# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN

def add_fcos_config(cfg):
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
    # FCOS Head
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.FCOS = CN()

    # This is the number of foreground classes.
    cfg.MODEL.FCOS.NUM_CLASSES = 80
    cfg.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.FCOS.FPN_FEAT_SIZE = 256

    # loss related
    cfg.MODEL.FCOS.OBJECT_SIZES_OF_INTEREST = [
        [-1, 64], # p3
        [64, 128], # p4
        [128, 256], # p5
        [256, 512], # p6
        [512, 9999999], # p7
    ]
    cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS = -1
    cfg.MODEL.FCOS.NORM_REG_TARGETS = False
    # test config
    cfg.MODEL.FCOS.TEST_SCORE_THRESH = 0.05
    cfg.MODEL.FCOS.NMS_THRESH_TEST = 0.5
    # input
    cfg.MODEL.FCOS.PIXEL_MEAN = [103.530, 116.280, 123.675]
    cfg.MODEL.FCOS.PIXEL_STD = [57.375, 57.120, 58.395]

    