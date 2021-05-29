# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
import numpy as np
from typing import List
import torch
from fvcore.nn import giou_loss, sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
# from detectron2.data.detection_utils import convert_image_to_rgb
# from detectron2.layers import ShapeSpec, batched_nms, cat, get_norm
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
# from ..anchor_generator import build_anchor_generator
from detectron2.modeling import build_backbone, detector_postprocess, META_ARCH_REGISTRY
from detectron2.utils.comm import get_world_size

from .box import Anchors, box_nms, calc_iou
import math
import torch.distributed as dist



__all__ = ['Retinanet2']

def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def reduce_mean(tensor):
    num_gpus = get_world_size()
    total = reduce_sum(tensor)
    return total.float() / num_gpus


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(num_features_in, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1) # bs h w c
        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(num_features_in, num_anchors * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out = out.permute(0, 2, 3, 1)
        out = out.contiguous().view(x.shape[0], -1, self.num_classes)
        return out

@META_ARCH_REGISTRY.register()
class Retinanet2(nn.Module):
    @configurable
    def __init__(self, 
        # structure
        backbone,
        head_in_features,
        fpn_feat_size,
        num_classes, 
        # anchor related
        anchor_sizes,
        anchor_ratios,
        anchor_scales,
        anchor_offset,
        # input related
        pixel_mean,
        pixel_std,
        # post process
        test_score_thresh=0.05,
        test_nms_thresh=0.5,
        bbox_reg_weight=[1,1,1,1],
    ):
        """
        NOTE: this interface is experimental.
        """
        super().__init__()

        self.backbone = backbone
        self.head_in_features = head_in_features
        self.num_classes = num_classes
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.bbox_reg_weight = bbox_reg_weight

        self.anchor_gen = Anchors(
            sizes=anchor_sizes,
            ratios=anchor_ratios,
            scales=anchor_scales,
            offset=anchor_offset,
        )
        self.regressionModel = RegressionModel(fpn_feat_size, num_anchors=self.anchor_gen.num_anchors)
        self.classificationModel = ClassificationModel(fpn_feat_size, num_classes=num_classes, num_anchors=self.anchor_gen.num_anchors)

        # init params
        for modules in [self.classificationModel, self.regressionModel]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        prior = 0.01
        torch.nn.init.normal_(self.classificationModel.output.weight, mean=0, std=0.01)
        # self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        torch.nn.init.normal_(self.regressionModel.output.weight, mean=0, std=0.01)
        # self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        # backbone_shape = backbone.output_shape()
        # feature_shapes = [backbone_shape[f] for f in cfg.MODEL.RETINANET.IN_FEATURES]
        # head = RetinaNetHead(cfg, feature_shapes)
        return {
            'backbone': backbone,
            "head_in_features": cfg.MODEL.RETINANET2.IN_FEATURES,
            "fpn_feat_size": cfg.MODEL.RETINANET2.FPN_FEAT_SIZE,
            'num_classes': cfg.MODEL.RETINANET2.NUM_CLASSES,
            'anchor_sizes': cfg.MODEL.RETINANET2.ANCHOR_SIZES,
            'anchor_ratios': cfg.MODEL.RETINANET2.ANCHOR_RATIOS,
            'anchor_scales': cfg.MODEL.RETINANET2.ANCHOR_SCALES,
            'anchor_offset': cfg.MODEL.RETINANET2.ANCHOR_OFFSET,
            'pixel_mean': cfg.MODEL.RETINANET2.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.RETINANET2.PIXEL_STD,
            'test_score_thresh': cfg.MODEL.RETINANET2.TEST_SCORE_THRESH,
            'test_nms_thresh': cfg.MODEL.RETINANET2.NMS_THRESH_TEST,
            'bbox_reg_weight': cfg.MODEL.RETINANET2.BBOX_REG_WEIGHT,
        }

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                
    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        B, _, H, W = images.tensor.shape
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]

        anchors = self.anchor_gen._get_anchor_boxes(input_size=(W, H), device=self.device, anchor_format='xyxy')

        loc_preds = torch.cat([self.regressionModel(f) for f in features], dim=1)
        cls_preds = torch.cat([self.classificationModel(f) for f in features], dim=1)
        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            anno_boxes = [x["instances"].gt_boxes.tensor.to(self.device) for x in batched_inputs]
            anno_labels = [x['instances'].gt_classes.to(self.device) for x in batched_inputs]
            # construct anno
            # max_num_anno = max([itm.shape[0] for itm in gt_boxes])
            # anno = torch.ones(len(batched_inputs), max_num_anno, 5, dtype=torch.float).to(self.device) * -1
            # for idx in range(len(batched_inputs)):
            #     gt_box = gt_boxes[idx]
            #     gt_label = gt_labels[idx]
            #     anno[idx, :gt_box.shape[0], :4] = gt_box
            #     anno[idx, :gt_box.shape[0], 4] = gt_label
            return self.losses(cls_preds, loc_preds, anchors, anno_boxes, anno_labels, features=features)
        else:
            anchors_xywh = Anchors.xyxy2xywh(anchors)
            results = []
            for idx in range(len(batched_inputs)):
                final_boxes, final_labels, final_scores = self.inference_single(loc_preds[idx], cls_preds[idx], anchors_xywh)
                result = Instances(images.image_sizes[idx])
                result.pred_boxes = Boxes(final_boxes)
                result.scores = final_scores
                result.pred_classes = final_labels
                results.append(result)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
    

    
    def inference_single(self, loc_preds, cls_preds, anchors):
        assert loc_preds.ndim == 2 and cls_preds.ndim == 2, 'shape: {}, {}'.format(loc_preds.shape, cls_preds.shape)
        boxes = self.anchor_gen.predict_transform(loc_preds, anchors, bbx_reg_weight=self.bbox_reg_weight)
        cls_preds = torch.sigmoid(cls_preds)
        scores, labels = cls_preds.max(1)
        # import ipdb;ipdb.set_trace()
        ids = torch.where(scores > self.test_score_thresh)[0]            # [#obj,]
        scores = scores[ids]
        labels = labels[ids]
        boxes = boxes[ids]

        final_labels = []
        final_boxes = []
        final_scores = []
        # iter over classes
        for i in range(cls_preds.shape[1]):
            ids = torch.where(labels == i)[0]
            if len(ids) == 0:
                continue
            tmp_labels = labels[ids]
            tmp_boxes = boxes[ids]
            tmp_scores = scores[ids]
            keep = box_nms(tmp_boxes, tmp_scores, threshold=self.test_nms_thresh)
            final_boxes.append(tmp_boxes[keep])
            final_labels.append(tmp_labels[keep])
            final_scores.append(tmp_scores[keep])
        if len(final_labels) == 0:
            final_boxes = torch.tensor([])
            final_scores = torch.tensor([])
            final_labels = torch.tensor([])
        else:
            final_labels = torch.cat(final_labels)
            final_scores = torch.cat(final_scores)
            final_boxes = torch.cat(final_boxes, dim=0)
        return final_boxes, final_labels, final_scores


    def losses(self, classifications, regressions, anchors, anno_boxes, anno_labels, **kargs):
        alpha = 0.25
        gamma = 2.0
        num_images = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor_widths  = (anchors[:, 2] - anchors[:, 0]).unsqueeze(0)
        anchor_heights = (anchors[:, 3] - anchors[:, 1]).unsqueeze(0)
        anchor_ctr_x   = 0.5 * (anchors[:, 0] + anchors[:, 2]).unsqueeze(0)
        anchor_ctr_y   = 0.5 * (anchors[:, 1] + anchors[:, 3]).unsqueeze(0)

        # match anno to anchors first
        with torch.no_grad():
            gt_labels = []
            gt_boxes = []
            for j in range(num_images):
                bbox_annotation = anno_boxes[j]
                anno_label = anno_labels[j]
                if bbox_annotation.shape[0] == 0:
                    gt_labels_j = torch.zeros(classifications.shape[1]).to(self.device) + self.num_classes # num_anchors
                    gt_boxes_j = torch.zeros_like(anchors) # num_anchors, 4
                else:
                    IoU = calc_iou(anchors, bbox_annotation) # num_anchors x num_annotations
                    IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1
                    
                    gt_labels_j = anno_label[IoU_argmax] # num_anchors
                    gt_boxes_j = bbox_annotation[IoU_argmax] # num_anchors, 4

                    # assign bg, ignore, pos based on IoU_max
                    # pos_mask = IoU_max >= 0.5
                    neg_mask = IoU_max <= 0.4
                    ignore_mask = (IoU_max > 0.4) & (IoU_max < 0.5)

                    gt_labels_j[neg_mask] = self.num_classes
                    gt_labels_j[ignore_mask] = -1

                
                gt_labels.append(gt_labels_j)
                gt_boxes.append(gt_boxes_j)
            
            gt_labels = torch.stack(gt_labels) # bs num_anchors
            gt_boxes = torch.stack(gt_boxes) # bs num_anchors 4

        # focal loss
        valid_mask = gt_labels >= 0
        
        valid_classifications = classifications[valid_mask] # num_valid * num_classes
        p = torch.sigmoid(valid_classifications)
        gt_labels_target = F.one_hot(gt_labels[valid_mask].long(), num_classes=self.num_classes + 1)[
            :, :-1
        ].to(valid_classifications.dtype)  # no loss for the last (background) class
        bce_loss = F.binary_cross_entropy_with_logits(valid_classifications, gt_labels_target, reduction="none")
        p_t = p * gt_labels_target + (1 - p) * (1 - gt_labels_target)
        focal_weight = (1 - p_t) ** gamma
        cls_loss = bce_loss * focal_weight
        alpha_weight = gt_labels_target * alpha + (1 - alpha) * (1 - gt_labels_target)
        cls_loss = cls_loss * alpha_weight
        cls_loss = cls_loss.sum()

        # regression loss
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes) # bs num_anchors
        num_pos_anchors = pos_mask.sum()
        denorm = max(reduce_mean(num_pos_anchors).item(), 1)
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors.item() / num_images)

        # transform gt to delta
        gt_ctr_x = 0.5 * (gt_boxes[:, :, 0] + gt_boxes[:, :, 2])
        gt_ctr_y = 0.5 * (gt_boxes[:, :, 1] + gt_boxes[:, :, 3])
        gt_width = gt_boxes[:, :, 2] - gt_boxes[:, :, 0]
        gt_height = gt_boxes[:, :, 3] - gt_boxes[:, :, 1]

        wx, wy, ww, wh = self.bbox_reg_weight
        dx = wx * (gt_ctr_x - anchor_ctr_x) / anchor_widths
        dy = wy * (gt_ctr_y - anchor_ctr_y) / anchor_heights
        dw = ww * torch.log(gt_width / anchor_widths)
        dh = wh * torch.log(gt_height / anchor_heights) # bs num_anchors
        # print('dx', dx)
        # print('dy', dy)
        # print('dw', dw)
        # print('dh', dh)

        deltas = torch.stack((dx, dy, dw, dh), dim=2) # bs num_anchors 4

        reg_loss = smooth_l1_loss(
            regressions[pos_mask],
            deltas[pos_mask],
            0.1,
            'sum'
        )
        return {
            'cls_loss': cls_loss / denorm,
            'reg_loss': reg_loss / denorm,
        }


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, size_divisibility=32)
        return images

