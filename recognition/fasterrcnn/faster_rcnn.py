import torch
import torch.nn as nn
import torchvision
import math
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_iou(boxes1, boxes2):
    """
    Compute the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1 (torch.Tensor): Shape (N, 4), bounding boxes [x1, y1, x2, y2].
        boxes2 (torch.Tensor): Shape (M, 4), bounding boxes [x1, y1, x2, y2].

    Returns:
        torch.Tensor: Shape (N, M), IoU values between boxes1 and boxes2.

    Notes:
        - IoU is 0 for non-overlapping boxes.
        - Inputs must be in [x1, y1, x2, y2] format.
    """

    # Area of boxes (x2-x1)*(y2-y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
    
    # Get top left x1,y1 coordinate
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # (N, M)
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # (N, M)
    
    # Get bottom right x2,y2 coordinate
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # (N, M)
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # (N, M)
    
    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)  # (N, M)
    union = area1[:, None] + area2 - intersection_area  # (N, M)
    iou = intersection_area / union  # (N, M)
    return iou


def boxes_to_transformation_targets(ground_truth_boxes, anchors_or_proposals):
    """
    Compute transformation targets (tx, ty, tw, th) for anchors/proposals.

    Args:
        ground_truth_boxes (torch.Tensor): Shape (N, 4), ground truth boxes [x1, y1, x2, y2].
        anchors_or_proposals (torch.Tensor): Shape (N, 4), anchor/proposal boxes [x1, y1, x2, y2].

    Returns:
        torch.Tensor: Shape (N, 4), transformation targets [tx, ty, tw, th].
    """
    
    # Get center_x,center_y,w,h from x1,y1,x2,y2 for anchors
    widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * widths
    center_y = anchors_or_proposals[:, 1] + 0.5 * heights
    
    # Get center_x,center_y,w,h from x1,y1,x2,y2 for gt boxes
    gt_widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
    gt_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    gt_center_x = ground_truth_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = ground_truth_boxes[:, 1] + 0.5 * gt_heights
    
    targets_dx = (gt_center_x - center_x) / widths
    targets_dy = (gt_center_y - center_y) / heights
    targets_dw = torch.log(gt_widths / widths)
    targets_dh = torch.log(gt_heights / heights)
    regression_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return regression_targets


def apply_regression_pred_to_anchors_or_proposals(box_transform_pred, anchors_or_proposals):
    """
    Apply regression predictions to anchors/proposals to generate predicted boxes.

    Args:
        box_transform_pred (torch.Tensor): Shape (N, C, 4), regression predictions 
                                           [dx, dy, dw, dh] for each class.
        anchors_or_proposals (torch.Tensor): Shape (N, 4), anchor/proposal boxes [x1, y1, x2, y2].

    Returns:
        torch.Tensor: Shape (N, C, 4), predicted boxes [x1, y1, x2, y2] for each class.
    """
    box_transform_pred = box_transform_pred.reshape(
        box_transform_pred.size(0), -1, 4)
    
    # Get cx, cy, w, h from x1,y1,x2,y2
    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * w
    center_y = anchors_or_proposals[:, 1] + 0.5 * h
    
    dx = box_transform_pred[..., 0]
    dy = box_transform_pred[..., 1]
    dw = box_transform_pred[..., 2]
    dh = box_transform_pred[..., 3]
    # dh -> (num_anchors_or_proposals, num_classes)
    
    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=math.log(1000.0 / 16))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16))
    
    pred_center_x = dx * w[:, None] + center_x[:, None]
    pred_center_y = dy * h[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * w[:, None]
    pred_h = torch.exp(dh) * h[:, None]
    # pred_center_x -> (num_anchors_or_proposals, num_classes)
    
    pred_box_x1 = pred_center_x - 0.5 * pred_w
    pred_box_y1 = pred_center_y - 0.5 * pred_h
    pred_box_x2 = pred_center_x + 0.5 * pred_w
    pred_box_y2 = pred_center_y + 0.5 * pred_h
    
    pred_boxes = torch.stack((
        pred_box_x1,
        pred_box_y1,
        pred_box_x2,
        pred_box_y2),
        dim=2)
    # pred_boxes -> (num_anchors_or_proposals, num_classes, 4)
    return pred_boxes


def sample_positive_negative(labels, positive_count, total_count):
    """
    Sample positive and negative indices based on the given labels.

    Args:
        labels (torch.Tensor): 1D tensor with labels where:
                               - `labels >= 1` are positive samples.
                               - `labels == 0` are negative samples.
        positive_count (int): Maximum number of positive samples to select.
        total_count (int): Total number of samples to select.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - sampled_neg_idx_mask (torch.Tensor): Boolean mask for selected negative samples.
            - sampled_pos_idx_mask (torch.Tensor): Boolean mask for selected positive samples.
    """
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]
    num_pos = positive_count
    num_pos = min(positive.numel(), num_pos)
    num_neg = total_count - num_pos
    num_neg = min(negative.numel(), num_neg)
    perm_positive_idxs = torch.randperm(positive.numel(),
                                        device=positive.device)[:num_pos]
    perm_negative_idxs = torch.randperm(negative.numel(),
                                        device=negative.device)[:num_neg]
    pos_idxs = positive[perm_positive_idxs]
    neg_idxs = negative[perm_negative_idxs]
    sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_pos_idx_mask[pos_idxs] = True
    sampled_neg_idx_mask[neg_idxs] = True
    return sampled_neg_idx_mask, sampled_pos_idx_mask


def clamp_boxes_to_image_boundary(boxes, image_shape):
    """
    Clamp bounding boxes to lie within image boundaries.

    Args:
        boxes (torch.Tensor): Tensor of shape (..., 4), where each box is [x1, y1, x2, y2].
        image_shape (Tuple[int, int]): Image dimensions as (height, width).

    Returns:
        torch.Tensor: Tensor of shape (..., 4) with clamped bounding boxes.
    """
    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]
    height, width = image_shape[-2:]
    boxes_x1 = boxes_x1.clamp(min=0, max=width)
    boxes_x2 = boxes_x2.clamp(min=0, max=width)
    boxes_y1 = boxes_y1.clamp(min=0, max=height)
    boxes_y2 = boxes_y2.clamp(min=0, max=height)
    boxes = torch.cat((
        boxes_x1[..., None],
        boxes_y1[..., None],
        boxes_x2[..., None],
        boxes_y2[..., None]),
        dim=-1)
    return boxes


def transform_boxes_to_original_size(boxes, new_size, original_size):
    """
    Transform bounding boxes from a resized image back to the original image size.

    Args:
        boxes (torch.Tensor): Tensor of shape (N, 4), bounding boxes [x1, y1, x2, y2].
        new_size (Tuple[int, int]): Size of the resized image as (height, width).
        original_size (Tuple[int, int]): Original image size as (height, width).

    Returns:
        torch.Tensor: Tensor of shape (N, 4), bounding boxes scaled to the original image size.
    """
    ratios = [
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class RegionProposalNetwork(nn.Module):
    """
    Implements a Region Proposal Network (RPN) for object detection.

    The RPN generates region proposals by predicting objectness scores and bounding box 
    transformations for anchors based on feature maps.

    Args:
        in_channels (int): Number of input channels.
        scales (list[float]): Anchor box scales (e.g., [128, 256, 512]).
        aspect_ratios (list[float]): Anchor box aspect ratios (e.g., [0.5, 1.0, 2.0]).
        model_config (dict): RPN configuration with thresholds, batch size, and NMS settings.

    Methods:
        - generate_anchors(image, feat): Creates anchors for the feature map.
        - assign_targets_to_anchors(anchors, gt_boxes): Assigns labels and targets to anchors.
        - filter_proposals(proposals, cls_scores, image_shape): Filters proposals using NMS.
        - forward(image, feat, target=None): Generates proposals and computes losses during training.

    Returns:
        - During inference: A dictionary with 'proposals' (bounding boxes) and 'scores' (objectness).
        - During training: Adds classification and localization losses.
    """
    
    def __init__(self, in_channels, scales, aspect_ratios, model_config):
        super(RegionProposalNetwork, self).__init__()
        self.scales = scales
        self.low_iou_threshold = model_config['rpn_bg_threshold']
        self.high_iou_threshold = model_config['rpn_fg_threshold']
        self.rpn_nms_threshold = model_config['rpn_nms_threshold']
        self.rpn_batch_size = model_config['rpn_batch_size']
        self.rpn_pos_count = int(model_config['rpn_pos_fraction'] * self.rpn_batch_size)
        self.rpn_topk = model_config['rpn_train_topk'] if self.training else model_config['rpn_test_topk']
        self.rpn_prenms_topk = model_config['rpn_train_prenms_topk'] if self.training \
            else model_config['rpn_test_prenms_topk']
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)
        
        # 3x3 conv layer
        self.rpn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        
        # 1x1 classification conv layer
        self.cls_layer = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1, stride=1)
        
        # 1x1 regression
        self.bbox_reg_layer = nn.Conv2d(in_channels, self.num_anchors * 4, kernel_size=1, stride=1)
        
        for layer in [self.rpn_conv, self.cls_layer, self.bbox_reg_layer]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
    
    def generate_anchors(self, image, feat):
        """
        Generate anchor boxes for the feature map.

        Args:
            image (torch.Tensor): Input image tensor of shape (N, C, H, W).
            feat (torch.Tensor): Feature map tensor of shape (N, C_feat, H_feat, W_feat).

        Returns:
            torch.Tensor: Anchor boxes of shape (H_feat * W_feat * num_anchors_per_location, 4).
        """
        grid_h, grid_w = feat.shape[-2:]
        image_h, image_w = image.shape[-2:]
        
        # For the vgg16 case stride would be 16 for both h and w
        stride_h = torch.tensor(image_h // grid_h, dtype=torch.int64, device=feat.device)
        stride_w = torch.tensor(image_w // grid_w, dtype=torch.int64, device=feat.device)
        
        scales = torch.as_tensor(self.scales, dtype=feat.dtype, device=feat.device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=feat.dtype, device=feat.device)
        
        # Assuming anchors of scale 128 sq pixels
        # For 1:1 it would be (128, 128) -> area=16384
        # For 2:1 it would be (181.02, 90.51) -> area=16384
        # For 1:2 it would be (90.51, 181.02) -> area=16384
        
        # The below code ensures h/w = aspect_ratios and h*w=1
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        
        # Now we will just multiply h and w with scale(example 128)
        # to make h*w = 128 sq pixels and h/w = aspect_ratios
        # This gives us the widths and heights of all anchors
        # which we need to replicate at all locations
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        
        # Now we make all anchors zero centred
        # So x1, y1, x2, y2 = -w/2, -h/2, w/2, h/2
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = base_anchors.round()
        
        # Get the shifts in x axis (0, 1,..., W_feat-1) * stride_w
        shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=feat.device) * stride_w

        # Get the shifts in x axis (0, 1,..., H_feat-1) * stride_h
        shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=feat.device) * stride_h
        
        # Create a grid using these shifts
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        # shifts_x -> (H_feat, W_feat)
        # shifts_y -> (H_feat, W_feat)
        
        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        # Setting shifts for x1 and x2(same as shifts_x) and y1 and y2(same as shifts_y)
        shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)
        # shifts -> (H_feat * W_feat, 4)
        
        # base_anchors -> (num_anchors_per_location, 4)
        # shifts -> (H_feat * W_feat, 4)
        # Add these shifts to each of the base anchors
        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))
        # anchors -> (H_feat * W_feat, num_anchors_per_location, 4)
        anchors = anchors.reshape(-1, 4)
        # anchors -> (H_feat * W_feat * num_anchors_per_location, 4)
        return anchors
    
    def assign_targets_to_anchors(self, anchors, gt_boxes):
        """
        Assign ground truth boxes to anchors based on IoU and generate classification labels.

        Args:
            anchors (torch.Tensor): Tensor of shape (num_anchors, 4), all anchor boxes.
            gt_boxes (torch.Tensor): Tensor of shape (num_gt_boxes, 4), ground truth boxes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - labels: Tensor of shape (num_anchors), values {-1, 0, 1} indicating:
                    - 1: Foreground (IoU > high threshold).
                    - 0: Background (IoU < low threshold).
                    - -1: Ignored (IoU between low and high thresholds).
                - matched_gt_boxes: Tensor of shape (num_anchors, 4), assigned ground truth boxes.
        """
        
        # Get (gt_boxes, num_anchors_in_image) IOU matrix
        iou_matrix = get_iou(gt_boxes, anchors)
        
        # For each anchor get the gt box index with maximum overlap
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        # best_match_gt_idx -> (num_anchors_in_image)
        
        # This copy of best_match_gt_idx will be needed later to
        # add low quality matches
        best_match_gt_idx_pre_thresholding = best_match_gt_idx.clone()
        
        # Based on threshold, update the values of best_match_gt_idx
        # For anchors with highest IOU < low_threshold update to be -1
        # For anchors with highest IOU between low_threshold & high threshold update to be -2
        below_low_threshold = best_match_iou < self.low_iou_threshold
        between_thresholds = (best_match_iou >= self.low_iou_threshold) & (best_match_iou < self.high_iou_threshold)
        best_match_gt_idx[below_low_threshold] = -1
        best_match_gt_idx[between_thresholds] = -2
        
        # Add low quality anchor boxes, if for a given ground truth box, these are the ones
        # that have highest IOU with that gt box
        
        # For each gt box, get the maximum IOU value amongst all anchors
        best_anchor_iou_for_gt, _ = iou_matrix.max(dim=1)
        # best_anchor_iou_for_gt -> (num_gt_boxes_in_image)
        
        # For each gt box get those anchors
        # which have this same IOU as present in best_anchor_iou_for_gt
        # This is to ensure if 10 anchors all have the same IOU value,
        # which is equal to the highest IOU that this gt box has with any anchor
        # then we get all these 10 anchors
        gt_pred_pair_with_highest_iou = torch.where(iou_matrix == best_anchor_iou_for_gt[:, None])
        # gt_pred_pair_with_highest_iou -> [0, 0, 0, 1, 1, 1], [8896,  8905,  8914, 10472, 10805, 11138]
        # This means that anchors at the first 3 indexes have an IOU with gt box at index 0
        # which is equal to the highest IOU that this gt box has with ANY anchor
        # Similarly anchor at last three indexes(10472, 10805, 11138) have an IOU with gt box at index 1
        # which is equal to the highest IOU that this gt box has with ANY anchor
        # These 6 anchor indexes will also be added as positive anchors
        
        # Get all the anchors indexes to update
        pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
        
        # Update the matched gt index for all these anchors with whatever was the best gt box
        # prior to thresholding
        best_match_gt_idx[pred_inds_to_update] = best_match_gt_idx_pre_thresholding[pred_inds_to_update]
        
        # best_match_gt_idx is either a valid index for all anchors or -1(background) or -2(to be ignored)
        # Clamp this so that the best_match_gt_idx is a valid non-negative index
        # At this moment the -1 and -2 labelled anchors will be mapped to the 0th gt box
        matched_gt_boxes = gt_boxes[best_match_gt_idx.clamp(min=0)]
        
        # Set all foreground anchor labels as 1
        labels = best_match_gt_idx >= 0
        labels = labels.to(dtype=torch.float32)
        
        # Set all background anchor labels as 0
        background_anchors = best_match_gt_idx == -1
        labels[background_anchors] = 0.0
        
        # Set all to be ignored anchor labels as -1
        ignored_anchors = best_match_gt_idx == -2
        labels[ignored_anchors] = -1.0
        # Later for classification we will only pick labels which have > 0 label
        
        return labels, matched_gt_boxes

    def filter_proposals(self, proposals, cls_scores, image_shape):
        """
        Filters proposals using pre-NMS and post-NMS top-k selection, boundary clamping, 
        and size constraints.

        Args:
            proposals (torch.Tensor): Shape (num_anchors, 4), proposed bounding boxes.
            cls_scores (torch.Tensor): Shape (num_anchors,), objectness scores (logits).
            image_shape (Tuple[int, int]): Resized image shape (height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - proposals: Filtered proposals of shape (num_filtered_proposals, 4).
                - cls_scores: Filtered objectness scores of shape (num_filtered_proposals).
        """
        # Pre NMS Filtering
        cls_scores = cls_scores.reshape(-1)
        cls_scores = torch.sigmoid(cls_scores)
        _, top_n_idx = cls_scores.topk(min(self.rpn_prenms_topk, len(cls_scores)))
        
        cls_scores = cls_scores[top_n_idx]
        proposals = proposals[top_n_idx]
        ##################
        
        # Clamp boxes to image boundary
        proposals = clamp_boxes_to_image_boundary(proposals, image_shape)
        ####################
        
        # Small boxes based on width and height filtering
        min_size = 16
        ws, hs = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        proposals = proposals[keep]
        cls_scores = cls_scores[keep]
        ####################
        
        # NMS based on objectness scores
        keep_mask = torch.zeros_like(cls_scores, dtype=torch.bool)
        keep_indices = torch.ops.torchvision.nms(proposals, cls_scores, self.rpn_nms_threshold)
        keep_mask[keep_indices] = True
        keep_indices = torch.where(keep_mask)[0]
        # Sort by objectness
        post_nms_keep_indices = keep_indices[cls_scores[keep_indices].sort(descending=True)[1]]
        
        # Post NMS topk filtering
        proposals, cls_scores = (proposals[post_nms_keep_indices[:self.rpn_topk]],
                                 cls_scores[post_nms_keep_indices[:self.rpn_topk]])
        
        return proposals, cls_scores
    
    def forward(self, image, feat, target=None):
        """
        Forward pass for the Region Proposal Network (RPN).

        Steps:
            1. Generate classification and box regression predictions.
            2. Create anchors and transform them to generate proposals.
            3. Filter proposals using NMS and other constraints.
            4. During training:
                - Assign ground truth labels and boxes to anchors.
                - Sample positive/negative anchors.
                - Compute classification and localization losses.

        Args:
            image (torch.Tensor): Input image tensor (N, C, H, W).
            feat (torch.Tensor): Feature map tensor from the backbone.
            target (dict, optional): Ground truth data for training.

        Returns:
            dict: 
                - In inference: {'proposals': proposals, 'scores': scores}.
                - In training: Also includes classification and localization losses.
        """
        # Call RPN layers
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        cls_scores = self.cls_layer(rpn_feat)
        box_transform_pred = self.bbox_reg_layer(rpn_feat)

        # Generate anchors
        anchors = self.generate_anchors(image, feat)
        
        # Reshape classification scores to be (Batch Size * H_feat * W_feat * Number of Anchors Per Location, 1)
        # cls_score -> (Batch_Size, Number of Anchors per location, H_feat, W_feat)
        number_of_anchors_per_location = cls_scores.size(1)
        cls_scores = cls_scores.permute(0, 2, 3, 1)
        cls_scores = cls_scores.reshape(-1, 1)
        # cls_score -> (Batch_Size*H_feat*W_feat*Number of Anchors per location, 1)
        
        # Reshape bbox predictions to be (Batch Size * H_feat * W_feat * Number of Anchors Per Location, 4)
        # box_transform_pred -> (Batch_Size, Number of Anchors per location*4, H_feat, W_feat)
        box_transform_pred = box_transform_pred.view(
            box_transform_pred.size(0),
            number_of_anchors_per_location,
            4,
            rpn_feat.shape[-2],
            rpn_feat.shape[-1])
        box_transform_pred = box_transform_pred.permute(0, 3, 4, 1, 2)
        box_transform_pred = box_transform_pred.reshape(-1, 4)
        # box_transform_pred -> (Batch_Size*H_feat*W_feat*Number of Anchors per location, 4)
        
        # Transform generated anchors according to box transformation prediction
        proposals = apply_regression_pred_to_anchors_or_proposals(
            box_transform_pred.detach().reshape(-1, 1, 4),
            anchors)
        proposals = proposals.reshape(proposals.size(0), 4)
        ######################
        
        proposals, scores = self.filter_proposals(proposals, cls_scores.detach(), image.shape)
        rpn_output = {
            'proposals': proposals,
            'scores': scores
        }
        if not self.training or target is None:
            # If we are not training no need to do anything
            return rpn_output
        else:
            # Assign gt box and label for each anchor
            labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets_to_anchors(
                anchors,
                target['bboxes'][0])
            
            # Based on gt assignment above, get regression target for the anchors
            # matched_gt_boxes_for_anchors -> (Number of anchors in image, 4)
            # anchors -> (Number of anchors in image, 4)
            regression_targets = boxes_to_transformation_targets(matched_gt_boxes_for_anchors, anchors)
            
            ####### Sampling positive and negative anchors ####
            # Our labels were {fg:1, bg:0, to_be_ignored:-1}
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
                labels_for_anchors,
                positive_count=self.rpn_pos_count,
                total_count=self.rpn_batch_size)
            
            sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
            
            localization_loss = (
                    torch.nn.functional.smooth_l1_loss(
                        box_transform_pred[sampled_pos_idx_mask],
                        regression_targets[sampled_pos_idx_mask],
                        beta=1 / 9,
                        reduction="sum",
                    )
                    / (sampled_idxs.numel())
            ) 

            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(cls_scores[sampled_idxs].flatten(),
                                                                            labels_for_anchors[sampled_idxs].flatten())

            rpn_output['rpn_classification_loss'] = cls_loss
            rpn_output['rpn_localization_loss'] = localization_loss
            return rpn_output


class ROIHead(nn.Module):
    """
    ROI Head for object detection, applied after ROI pooling.

    Structure:
        - Two fully connected (fc) layers.
        - A classification layer for class scores.
        - A bounding box regression layer for box transformations.

    Args:
        model_config (dict): Configuration with ROI-specific settings (batch size, thresholds, etc.).
        num_classes (int): Number of classes for classification.
        in_channels (int): Input channels from ROI pooling.

    Attributes:
        - `fc6` and `fc7`: Fully connected layers for feature transformation.
        - `cls_layer`: Outputs class scores.
        - `bbox_reg_layer`: Outputs bounding box adjustments.
    """
    def __init__(self, model_config, num_classes, in_channels):
        super(ROIHead, self).__init__()
        self.num_classes = num_classes
        self.roi_batch_size = model_config['roi_batch_size']
        self.roi_pos_count = int(model_config['roi_pos_fraction'] * self.roi_batch_size)
        self.iou_threshold = model_config['roi_iou_threshold']
        self.low_bg_iou = model_config['roi_low_bg_iou']
        self.nms_threshold = model_config['roi_nms_threshold']
        self.topK_detections = model_config['roi_topk_detections']
        self.low_score_threshold = model_config['roi_score_threshold']
        self.pool_size = model_config['roi_pool_size']
        self.fc_inner_dim = model_config['fc_inner_dim']
        
        self.fc6 = nn.Linear(in_channels * self.pool_size * self.pool_size, self.fc_inner_dim)
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)
        self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)
        
        torch.nn.init.normal_(self.cls_layer.weight, std=0.01)
        torch.nn.init.constant_(self.cls_layer.bias, 0)

        torch.nn.init.normal_(self.bbox_reg_layer.weight, std=0.001)
        torch.nn.init.constant_(self.bbox_reg_layer.bias, 0)
    
    def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):
        """
        Assign ground truth boxes and labels to proposals using IOU.

        Args:
            proposals (torch.Tensor): Proposed boxes, shape (num_proposals, 4).
            gt_boxes (torch.Tensor): Ground truth boxes, shape (num_gt_boxes, 4).
            gt_labels (torch.Tensor): Labels for ground truth boxes, shape (num_gt_boxes,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - labels: Class labels for proposals (0: background, -1: ignored, positive: classes).
                - matched_gt_boxes: Assigned ground truth boxes, shape (num_proposals, 4).
        """
        # Get IOU Matrix between gt boxes and proposals
        iou_matrix = get_iou(gt_boxes, proposals)
        # For each gt box proposal find best matching gt box
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        background_proposals = (best_match_iou < self.iou_threshold) & (best_match_iou >= self.low_bg_iou)
        ignored_proposals = best_match_iou < self.low_bg_iou
        
        # Update best match of low IOU proposals to -1
        best_match_gt_idx[background_proposals] = -1
        best_match_gt_idx[ignored_proposals] = -2
        
        # Get best marching gt boxes for ALL proposals
        # Even background proposals would have a gt box assigned to it
        # Label will be used to ignore them later
        matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)]
        
        # Get class label for all proposals according to matching gt boxes
        labels = gt_labels[best_match_gt_idx.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)
        
        # Update background proposals to be of label 0(background)
        labels[background_proposals] = 0
        
        # Set all to be ignored anchor labels as -1(will be ignored)
        labels[ignored_proposals] = -1
        
        return labels, matched_gt_boxes_for_proposals
    
    def forward(self, feat, proposals, image_shape, target):
        """
        Forward pass for the ROI Head.

        Steps:
            1. Assign targets and sample proposals (training only).
            2. Perform ROI pooling on proposals.
            3. Process features through fully connected layers for classification and regression.
            4. Compute losses (training) or generate final predictions (inference).

        Args:
            feat (torch.Tensor): Feature map tensor.
            proposals (torch.Tensor): Proposals, shape (N, 4).
            image_shape (Tuple[int, int]): Image dimensions (H, W).
            target (dict, optional): Ground truth data for training.

        Returns:
            dict:
                - Training: Classification and localization losses.
                - Inference: Predicted boxes, scores, and labels.
        """
        if self.training and target is not None:
            # Add ground truth to proposals
            proposals = torch.cat([proposals, target['bboxes'][0]], dim=0)
            
            gt_boxes = target['bboxes'][0]
            gt_labels = target['labels'][0]
            
            labels, matched_gt_boxes_for_proposals = self.assign_target_to_proposals(proposals, gt_boxes, gt_labels)
            
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(labels,
                                                                                  positive_count=self.roi_pos_count,
                                                                                  total_count=self.roi_batch_size)
            
            sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
            
            # Keep only sampled proposals
            proposals = proposals[sampled_idxs]
            labels = labels[sampled_idxs]
            matched_gt_boxes_for_proposals = matched_gt_boxes_for_proposals[sampled_idxs]
            regression_targets = boxes_to_transformation_targets(matched_gt_boxes_for_proposals, proposals)
            # regression_targets -> (sampled_training_proposals, 4)
            # matched_gt_boxes_for_proposals -> (sampled_training_proposals, 4)
        
        # Get desired scale to pass to roi pooling function
        # For vgg16 case this would be 1/16 (0.0625)
        size = feat.shape[-2:]
        possible_scales = []
        for s1, s2 in zip(size, image_shape):
            approx_scale = float(s1) / float(s2)
            scale = 2 ** float(torch.tensor(approx_scale).log2().round())
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        
        # ROI pooling and call all layers for prediction
        proposal_roi_pool_feats = torchvision.ops.roi_pool(feat, [proposals],
                                                           output_size=self.pool_size,
                                                           spatial_scale=possible_scales[0])
        proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)
        box_fc_6 = torch.nn.functional.relu(self.fc6(proposal_roi_pool_feats))
        box_fc_7 = torch.nn.functional.relu(self.fc7(box_fc_6))
        cls_scores = self.cls_layer(box_fc_7)
        box_transform_pred = self.bbox_reg_layer(box_fc_7)
        # cls_scores -> (proposals, num_classes)
        # box_transform_pred -> (proposals, num_classes * 4)
        ##############################################
        
        num_boxes, num_classes = cls_scores.shape
        box_transform_pred = box_transform_pred.reshape(num_boxes, num_classes, 4)
        frcnn_output = {}
        if self.training and target is not None:
            classification_loss = torch.nn.functional.cross_entropy(cls_scores, labels)
            
            # Compute localization loss only for non-background labelled proposals
            fg_proposals_idxs = torch.where(labels > 0)[0]
            # Get class labels for these positive proposals
            fg_cls_labels = labels[fg_proposals_idxs]
            
            localization_loss = torch.nn.functional.smooth_l1_loss(
                box_transform_pred[fg_proposals_idxs, fg_cls_labels],
                regression_targets[fg_proposals_idxs],
                beta=1/9,
                reduction="sum",
            )
            localization_loss = localization_loss / labels.numel()
            frcnn_output['frcnn_classification_loss'] = classification_loss
            frcnn_output['frcnn_localization_loss'] = localization_loss
        
        if self.training:
            return frcnn_output
        else:
            device = cls_scores.device
            # Apply transformation predictions to proposals
            pred_boxes = apply_regression_pred_to_anchors_or_proposals(box_transform_pred, proposals)
            pred_scores = torch.nn.functional.softmax(cls_scores, dim=-1)
            
            # Clamp box to image boundary
            pred_boxes = clamp_boxes_to_image_boundary(pred_boxes, image_shape)
            
            # create labels for each prediction
            pred_labels = torch.arange(num_classes, device=device)
            pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)
            
            # remove predictions with the background label
            pred_boxes = pred_boxes[:, 1:]
            pred_scores = pred_scores[:, 1:]
            pred_labels = pred_labels[:, 1:]
            
            # pred_boxes -> (number_proposals, num_classes-1, 4)
            # pred_scores -> (number_proposals, num_classes-1)
            # pred_labels -> (number_proposals, num_classes-1)
            
            # batch everything, by making every class prediction be a separate instance
            pred_boxes = pred_boxes.reshape(-1, 4)
            pred_scores = pred_scores.reshape(-1)
            pred_labels = pred_labels.reshape(-1)
            
            pred_boxes, pred_labels, pred_scores = self.filter_predictions(pred_boxes, pred_labels, pred_scores)
            frcnn_output['boxes'] = pred_boxes
            frcnn_output['scores'] = pred_scores
            frcnn_output['labels'] = pred_labels
            return frcnn_output
    
    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        """
        Filters predictions by:
            1. Removing boxes with low scores.
            2. Discarding boxes below a minimum size.
            3. Applying class-wise Non-Maximum Suppression (NMS).
            4. Retaining only top-K detections.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes, shape (N, 4).
            pred_labels (torch.Tensor): Predicted labels, shape (N,).
            pred_scores (torch.Tensor): Confidence scores, shape (N,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Filtered boxes, shape (M, 4).
                - Filtered labels, shape (M,).
                - Filtered scores, shape (M,).
        """
        # remove low scoring boxes
        keep = torch.where(pred_scores > self.low_score_threshold)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        
        # Remove small boxes
        min_size = 16
        ws, hs = pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        
        # Class wise nms
        keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
        for class_id in torch.unique(pred_labels):
            curr_indices = torch.where(pred_labels == class_id)[0]
            curr_keep_indices = torch.ops.torchvision.nms(pred_boxes[curr_indices],
                                                          pred_scores[curr_indices],
                                                          self.nms_threshold)
            keep_mask[curr_indices[curr_keep_indices]] = True
        keep_indices = torch.where(keep_mask)[0]
        post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(descending=True)[1]]
        keep = post_nms_keep_indices[:self.topK_detections]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        return pred_boxes, pred_labels, pred_scores


class FasterRCNN(nn.Module):
    """
    Implements the Faster R-CNN object detection model with a VGG16 backbone.

    Args:
        model_config (dict): Configuration parameters for the model, including:
            - 'backbone_out_channels' (int): Number of output channels from the backbone.
            - 'scales' (list): Anchor box scales.
            - 'aspect_ratios' (list): Anchor box aspect ratios.
            - 'min_im_size' (int): Minimum image size for resizing.
            - 'max_im_size' (int): Maximum image size for resizing.
        num_classes (int): Number of object classes to detect (including background).

    Attributes:
        backbone (nn.Module): Feature extractor based on VGG16.
        rpn (RegionProposalNetwork): Generates region proposals from backbone features.
        roi_head (ROIHead): Performs classification and bounding box regression on proposals.
        image_mean (list): Mean values for image normalization.
        image_std (list): Standard deviation values for image normalization.
        min_size (int): Minimum image size for resizing.
        max_size (int): Maximum image size for resizing.

    Methods:
        normalize_resize_image_and_boxes(image, bboxes):
            Normalizes and resizes the input image and adjusts bounding boxes accordingly.

        forward(image, target=None):
            Forward pass through the model. If training, processes the target annotations.
            Returns region proposals and final detections or losses, depending on the mode.
    """
    def __init__(self, model_config, num_classes):
        super(FasterRCNN, self).__init__()
        self.model_config = model_config
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.backbone = vgg16.features[:-1]
        self.rpn = RegionProposalNetwork(model_config['backbone_out_channels'],
                                         scales=model_config['scales'],
                                         aspect_ratios=model_config['aspect_ratios'],
                                         model_config=model_config)
        self.roi_head = ROIHead(model_config, num_classes, in_channels=model_config['backbone_out_channels'])
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = model_config['min_im_size']
        self.max_size = model_config['max_im_size']
    
    def normalize_resize_image_and_boxes(self, image, bboxes):
        """
        Normalize and resize the input image and adjust bounding boxes accordingly.

        This function performs the following steps:
        1. **Normalization**: Adjusts the image pixel values to have a mean and standard deviation
        specified by `self.image_mean` and `self.image_std`.
        2. **Resizing**: Scales the image so that its smaller dimension matches `self.min_size` and
        the larger dimension does not exceed `self.max_size`, maintaining the aspect ratio.
        3. **Bounding Box Adjustment**: If bounding boxes are provided, their coordinates are scaled
        to match the resized image dimensions.

        Args:
            image (torch.Tensor): The input image tensor of shape (C, H, W).
            bboxes (torch.Tensor, optional): Tensor of bounding boxes with shape (N, 4), where N is
                the number of boxes, and each box is represented as (xmin, ymin, xmax, ymax).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The normalized and resized image tensor.
                - The adjusted bounding boxes tensor with the same shape as the input `bboxes`.
                If `bboxes` is None, returns None for the bounding boxes.
        """
        dtype, device = image.dtype, image.device
        
        # Normalize
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        image = (image - mean[:, None, None]) / std[:, None, None]
        #############
        
        # Resize to 1000x600 such that lowest size dimension is scaled upto 600
        # but larger dimension is not more than 1000
        # So compute scale factor for both and scale is minimum of these two
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(float(self.min_size) / min_size, float(self.max_size) / max_size)
        scale_factor = scale.item()
        
        # Resize image based on scale computed
        image = torch.nn.functional.interpolate(
            image,
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=True,
            align_corners=False,
        )

        if bboxes is not None:
            # Resize boxes by
            ratios = [
                torch.tensor(s, dtype=torch.float32, device=bboxes.device)
                / torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                for s, s_orig in zip(image.shape[-2:], (h, w))
            ]
            ratio_height, ratio_width = ratios
            xmin, ymin, xmax, ymax = bboxes.unbind(2)
            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height
            bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
        return image, bboxes
    
    def forward(self, image, target=None):
        """
        Forward pass for the Faster R-CNN model.

        Args:
            image (Tensor): Input image tensor of shape (C, H, W).
            target (dict, optional): Ground truth data containing:
                - 'bboxes' (Tensor): Bounding boxes of shape (N, 4).
                - 'labels' (Tensor): Class labels of shape (N,).

        Returns:
            tuple: Depending on the mode (training or inference):
                - During training:
                    - rpn_output (dict): Contains RPN losses.
                    - frcnn_output (dict): Contains Fast R-CNN losses.
                - During inference:
                    - rpn_output (dict): Contains proposals and scores.
                    - frcnn_output (dict): Contains detected boxes, scores, and labels.
        """
        old_shape = image.shape[-2:]
        if self.training:
            # Normalize and resize boxes
            image, bboxes = self.normalize_resize_image_and_boxes(image, target['bboxes'])
            target['bboxes'] = bboxes
        else:
            image, _ = self.normalize_resize_image_and_boxes(image, None)
        
        # Call backbone
        feat = self.backbone(image)
        
        # Call RPN and get proposals
        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output['proposals']
        
        # Call ROI head and convert proposals to boxes
        frcnn_output = self.roi_head(feat, proposals, image.shape[-2:], target)
        if not self.training:
            # Transform boxes to original image dimensions called only during inference
            frcnn_output['boxes'] = transform_boxes_to_original_size(frcnn_output['boxes'],
                                                                     image.shape[-2:],
                                                                     old_shape)
        return rpn_output, frcnn_output