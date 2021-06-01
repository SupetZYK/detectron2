# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
import numpy as np
from typing import Dict, List, Tuple
import torch
from fvcore.nn import giou_loss, sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
# from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat, get_norm, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
# from ..anchor_generator import build_anchor_generator
from detectron2.modeling import build_backbone, detector_postprocess, META_ARCH_REGISTRY
from detectron2.utils.comm import get_world_size

import math
import torch.distributed as dist
from .fcos_utils import FCOSLabelTarget, iou_loss


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
    def __init__(self, num_features_in):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(num_features_in, 4, kernel_size=3, padding=1)

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
    def __init__(self, num_features_in, num_classes=80):
        super(ClassificationModel, self).__init__()

        self.nc = num_classes

        self.conv1 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(num_features_in, self.nc, kernel_size=3, padding=1)

        self.centerness = nn.Conv2d(num_features_in, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        feat = self.act4(out)

        out = self.output(feat)

        out = out.permute(0, 2, 3, 1) # B H W C
        out = out.view(x.shape[0], -1, self.nc).contiguous()
        centerness = self.centerness(feat) # B 1 H W
        return out, centerness.view(x.shape[0], -1, 1)

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

@META_ARCH_REGISTRY.register()
class FCOS(nn.Module):
    @configurable
    def __init__(self, 
        # structure
        backbone,
        head_in_features,
        fpn_feat_size,
        num_classes, 
        # fcos related
        object_sizes_of_interest,
        center_sampling_radius,
        norm_reg_targets,
        # input related
        pixel_mean,
        pixel_std,
        # post process
        test_score_thresh=0.05,
        test_nms_thresh=0.5,
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

        self.regressionModel = RegressionModel(fpn_feat_size)
        self.classificationModel = ClassificationModel(fpn_feat_size, num_classes=num_classes)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(self.head_in_features))])
        self.strides = [2 ** (3+i) for i in range(len(self.head_in_features))] # hard code here
        self.fcos_target = FCOSLabelTarget(
            self.strides, 
            object_sizes_of_interest, 
            self.num_classes,
            center_sampling_radius=center_sampling_radius,
            norm_reg_targets=norm_reg_targets,
        )

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
            "head_in_features": cfg.MODEL.FCOS.IN_FEATURES,
            "fpn_feat_size": cfg.MODEL.FCOS.FPN_FEAT_SIZE,
            'num_classes': cfg.MODEL.FCOS.NUM_CLASSES,
            'object_sizes_of_interest': cfg.MODEL.FCOS.OBJECT_SIZES_OF_INTEREST,
            'center_sampling_radius': cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS,
            'norm_reg_targets': cfg.MODEL.FCOS.NORM_REG_TARGETS,
            'pixel_mean': cfg.MODEL.FCOS.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.FCOS.PIXEL_STD,
            'test_score_thresh': cfg.MODEL.FCOS.TEST_SCORE_THRESH,
            'test_nms_thresh': cfg.MODEL.FCOS.NMS_THRESH_TEST,
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

        locations = self.compute_locations(features)

        loc_preds = [torch.exp(self.scales[i](self.regressionModel(f))) for i, f in enumerate(features)]
        cls_preds = []
        centerness = []
        for f in features:
            cls_pred, cent = self.classificationModel(f)
            cls_preds.append(cls_pred)
            centerness.append(cent)

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            targets = []
            for x in batched_inputs:
                targets.append(
                    torch.cat([x["instances"].gt_classes.view(-1, 1), x["instances"].gt_boxes.tensor], dim=1).to(self.device)
                )
            # anno_boxes = [x["instances"].gt_boxes.tensor.to(self.device) for x in batched_inputs]
            # anno_labels = [x['instances'].gt_classes.to(self.device) for x in batched_inputs]
            return self.losses(locations, cls_preds, loc_preds, centerness, targets)
        else:
            results = []
            for idx in range(len(batched_inputs)):
                per_img_cls_preds = [cls_pred[idx] for cls_pred in cls_preds]
                per_img_loc_preds = [loc_pred[idx] for loc_pred in loc_preds]
                per_img_cent_preds = [cent[idx] for cent in centerness]
                locations = [loc.view(-1, 2) for loc in locations]
                results_per_img = self.inference_single(locations, per_img_cls_preds, per_img_loc_preds, per_img_cent_preds, images.image_sizes[idx])
                results.append(results_per_img)
            # post process
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
    
    def compute_locations(self, feature_maps):
        all_locations = []
        for level, feat in enumerate(feature_maps):
            _, _, ny, nx = feat.shape
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            locations = torch.stack([xv, yv], dim=2).view(1, -1, 2).to(feat.device)
            locations = locations * self.strides[level] + self.strides[level] // 2
            all_locations.append(locations)
        return all_locations


    def inference_single(
        self,
        locations: List[Tensor],
        box_cls: List[Tensor],
        box_delta: List[Tensor],
        cent_pred: List[Tensor],
        image_size: Tuple[int, int],
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, cent_i, loc_i in zip(box_cls, box_delta, cent_pred, locations):
            # (HxWxAxK,)
            predicted_prob = box_cls_i.sigmoid() * cent_i.sigmoid()
            predicted_prob = predicted_prob.flatten()
            # Apply two filtering below to make NMS faster.
            # 1. Keep boxes with confidence score higher than threshold
            keep_idxs = predicted_prob > self.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            # 2. Keep top k top scoring boxes only
            # num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            num_topk = topk_idxs.size(0)
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            loc_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes
            box_reg_i = box_reg_i[loc_idxs]
            loc_i = loc_i[loc_idxs]
            # predict boxes
            xy1_ = loc_i - box_reg_i[:,:2]
            xy2_ = loc_i + box_reg_i[:,2:]
            predicted_boxes = torch.cat([xy1_, xy2_], dim=-1)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.test_nms_thresh)
        # keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def losses(self, locations, classifications, regressions, centerness, targets, **kargs):
        """[summary]

        Args:
            locations (List(tensor)): grid points
            classifications (List(tensor)): each tensor for a prediction of a single feature map
            regressions (List(tensor)): [description]
            grids (List(tensor)): [description]
            targets (tensor): [description]

        Returns:
            [dict]: loss dict
        """
        alpha = 0.25
        gamma = 2.0

        # reformat targets to List targets
        num_images = classifications[0].shape[0]
        # prepare taregts, labels: List of tensor(bs, np); reg_targets: List of tensor(bs, np, 4)
        with torch.no_grad():
            labels, reg_targets = self.fcos_target.prepare_targets(locations, targets)

        # cat all labels and reg_targets
        labels = torch.cat(labels, -1)
        reg_targets = torch.cat(reg_targets, 1)
        valid_mask = labels >= 0
        pos_mask = valid_mask & (labels < self.num_classes) 

        # statistics
        num_pos = max(reduce_mean(pos_mask.sum()).item(), 1.0)

        # cat all cls_preds, loc_preds and centerness
        classifications = torch.cat(classifications, 1) #(bs, n, nc)
        regressions = torch.cat(regressions, 1) # (bs, n, 4)
        centerness = torch.cat(centerness, 1) # (bs, n, 1)

        # cls loss
        valid_classifications = classifications[valid_mask]
        p = torch.sigmoid(valid_classifications)
        gt_labels_target = F.one_hot(labels[valid_mask].long(), num_classes=self.num_classes + 1)[
            :, :-1
        ].to(valid_classifications.dtype)  # no loss for the last (background) class
        bce_loss = F.binary_cross_entropy_with_logits(valid_classifications, gt_labels_target, reduction="none")
        p_t = p * gt_labels_target + (1 - p) * (1 - gt_labels_target)
        focal_weight = (1 - p_t) ** gamma
        cls_loss = bce_loss * focal_weight
        alpha_weight = gt_labels_target * alpha + (1 - alpha) * (1 - gt_labels_target)
        cls_loss = cls_loss * alpha_weight
        cls_loss = cls_loss.sum() / num_pos

        # centerness loss
        def compute_centerness_targets(reg_targets):
            if reg_targets.shape[0] == 0:
                return torch.ones(0).to(reg_targets.device)
            left_right = reg_targets[:, [0, 2]]
            top_bottom = reg_targets[:, [1, 3]]
            centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                        (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
            return torch.sqrt(centerness)

        
        with torch.no_grad():
            masked_reg_targets = reg_targets[pos_mask]
            centerness_target = compute_centerness_targets(masked_reg_targets)
        masked_centerness = centerness[pos_mask].view(-1)
        center_loss = F.binary_cross_entropy_with_logits(masked_centerness, centerness_target, reduction='sum') / num_pos
        # reg loss
        denorm = max(reduce_mean(centerness_target.sum()).item(), 1e-6)
        reg_loss = iou_loss(regressions[pos_mask], masked_reg_targets, centerness_target, loss_type='giou') / denorm
        # if not torch.isfinite(reg_loss):
        #     import ipdb;ipdb.set_trace()
        #     print('err')
        return {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'center_loss': center_loss,
        }

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, size_divisibility=32)
        return images

