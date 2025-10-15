from typing import *
import math

import torch
import torch.nn.functional as F
import utils3d

from ..utils.geometry_torch import (
    weighted_mean, 
    harmonic_mean, 
    geometric_mean,
    mask_aware_nearest_resize,
    normalized_view_plane_uv,
    angle_diff_vec3
)
from ..utils.alignment import (
    align_points_scale_z_shift, 
    align_points_scale, 
    align_points_scale_xyz_shift,
    align_points_z_shift,
)


def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def affine_invariant_global_loss(
    pred_points: torch.Tensor, 
    gt_points: torch.Tensor, 
    mask: torch.Tensor, 
    align_resolution: int = 64, 
    beta: float = 0.0, 
    trunc: float = 1.0, 
    sparsity_aware: bool = False,
    gt_metric_scale: torch.Tensor = None,
    gt_shift: torch.Tensor = None
):
    device = pred_points.device

    if gt_metric_scale is not None and gt_shift is not None:
        scale = gt_metric_scale
        shift = gt_shift
        valid = scale > 0
        scale, shift = torch.where(valid, scale, 0), torch.where(valid[..., None], shift, 0)
    else:
        if pred_points.dim() == 3:
            # Align
            (pred_points_lr, gt_points_lr), lr_mask = mask_aware_nearest_resize((pred_points, gt_points), mask=mask, size=(align_resolution, align_resolution))
            scale, shift = align_points_scale_z_shift(pred_points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2), lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=trunc)
            valid = scale > 0
            scale, shift = torch.where(valid, scale, 0), torch.where(valid[..., None], shift, 0)
        elif pred_points.dim() == 4:
            pred_points_list, gt_points_list, mask_list = [], [], []
            for pred_points_i, gt_points_i, mask_i in zip(pred_points, gt_points, mask):
                # Align
                (pred_points_lr, gt_points_lr), lr_mask = mask_aware_nearest_resize((pred_points_i, gt_points_i), mask=mask_i, size=(align_resolution, align_resolution))
                pred_points_list.append(pred_points_lr)
                gt_points_list.append(gt_points_lr)
                mask_list.append(lr_mask)
            pred_points_lr = torch.cat(pred_points_list, dim=0)
            gt_points_lr = torch.cat(gt_points_list, dim=0)
            lr_mask = torch.cat(mask_list, dim=0)
            scale, shift = align_points_scale_z_shift(pred_points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2), lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=trunc)
            valid = scale > 0
            scale, shift = torch.where(valid, scale, 0), torch.where(valid[..., None], shift, 0)
            return scale.detach(), shift.detach()
        else:
            raise ValueError(f'Invalid pred_points dimension: {pred_points.dim()}')

    pred_points = scale[..., None, None, None] * pred_points + shift[..., None, None, :]

    # Compute loss
    weight = (valid[..., None, None] & mask).float() / gt_points[..., 2].clamp_min(1e-5)
    weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))   # In case your data contains extremely small depth values
    loss = _smooth((pred_points - gt_points).abs() * weight[..., None], beta=beta).mean(dim=(-3, -2, -1))

    if sparsity_aware:
        # Reweighting improves performance on sparse depth data. NOTE: this is not used in MoGe-1.
        sparsity = mask.float().mean(dim=(-2, -1)) / lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)

    err = (pred_points.detach() - gt_points).norm(dim=-1) / gt_points[..., 2]

    # Record any scalar metric
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), mask).item(),
        'delta': weighted_mean((err < 1).float(), mask).item()
    }

    return loss, misc, scale.detach(), shift.detach()


def monitoring(points: torch.Tensor):
    return {
        'std': points.std().item(),
    }


def compute_anchor_sampling_weight(
    points: torch.Tensor, 
    mask: torch.Tensor, 
    radius_2d: torch.Tensor, 
    radius_3d: torch.Tensor, 
    num_test: int = 64
) -> torch.Tensor:
    # Importance sampling to balance the sampled probability of fine strutures.
    # NOTE: MoGe-1 uses uniform random sampling instead of importance sampling.
    #       This is an incremental trick introduced later than the publication of MoGe-1 paper.

    height, width = points.shape[-3:-1]

    pixel_i, pixel_j = torch.meshgrid(
        torch.arange(height, device=points.device), 
        torch.arange(width, device=points.device),
        indexing='ij'
    )
    
    test_delta_i = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_delta_j = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_i, test_j = pixel_i[..., None] + test_delta_i, pixel_j[..., None] + test_delta_j                       # [height, width, num_test]
    test_mask = (test_i >= 0) & (test_i < height) & (test_j >= 0) & (test_j < width)                            # [height, width, num_test]
    test_i, test_j = test_i.clamp(0, height - 1), test_j.clamp(0, width - 1)                                    # [height, width, num_test]
    test_mask = test_mask & mask[..., test_i, test_j]                                                           # [..., height, width, num_test]
    test_points = points[..., test_i, test_j, :]                                                                # [..., height, width, num_test, 3]
    test_dist = (test_points - points[..., None, :]).norm(dim=-1)                                               # [..., height, width, num_test]

    weight = 1 / ((test_dist <= radius_3d[..., None]) & test_mask).float().sum(dim=-1).clamp_min(1)
    weight = torch.where(mask, weight, 0)
    weight = weight / weight.sum(dim=(-2, -1), keepdim=True).add(1e-7)                                          # [..., height, width]
    return weight


def affine_invariant_local_loss(
    pred_points: torch.Tensor, 
    gt_points: torch.Tensor, 
    gt_mask: torch.Tensor, 
    focal: torch.Tensor, 
    global_scale: torch.Tensor, 
    level: Literal[4, 16, 64], 
    align_resolution: int = 32, 
    num_patches: int = 16, 
    beta: float = 0.0, 
    trunc: float = 1.0, 
    sparsity_aware: bool = False
):
    device, dtype = pred_points.device, pred_points.dtype
    *batch_shape, height, width, _ = pred_points.shape
    batch_size = math.prod(batch_shape)
    pred_points, gt_points, gt_mask, focal, global_scale = pred_points.reshape(-1, height, width, 3), gt_points.reshape(-1, height, width, 3), gt_mask.reshape(-1, height, width), focal.reshape(-1), global_scale.reshape(-1) if global_scale is not None else None
    
    # Sample patch anchor points indices [num_total_patches]
    radius_2d = math.ceil(0.5 / level * (height ** 2 + width ** 2) ** 0.5)
    radius_3d = 0.5 / level / focal * gt_points[..., 2]
    anchor_sampling_weights = compute_anchor_sampling_weight(gt_points, gt_mask, radius_2d, radius_3d, num_test=64)
    where_mask = torch.where(gt_mask)
    random_selection = torch.multinomial(anchor_sampling_weights[where_mask], num_patches * batch_size, replacement=True)
    patch_batch_idx, patch_anchor_i, patch_anchor_j = [indices[random_selection] for indices in where_mask]     # [num_total_patches]

    # Get patch indices [num_total_patches, patch_h, patch_w]
    patch_i, patch_j = torch.meshgrid(
        torch.arange(-radius_2d, radius_2d + 1, device=device), 
        torch.arange(-radius_2d, radius_2d + 1, device=device),
        indexing='ij'
    )
    patch_i, patch_j = patch_i + patch_anchor_i[:, None, None], patch_j + patch_anchor_j[:, None, None]
    patch_mask = (patch_i >= 0) & (patch_i < height) & (patch_j >= 0) & (patch_j < width)
    patch_i, patch_j = patch_i.clamp(0, height - 1), patch_j.clamp(0, width - 1)
    
    # Get patch mask and gt patch points
    gt_patch_anchor_points = gt_points[patch_batch_idx, patch_anchor_i, patch_anchor_j]
    gt_patch_radius_3d = 0.5 / level / focal[patch_batch_idx] * gt_patch_anchor_points[:, 2]
    gt_patch_points = gt_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    gt_patch_dist = (gt_patch_points - gt_patch_anchor_points[:, None, None, :]).norm(dim=-1)    
    patch_mask &= gt_mask[patch_batch_idx[:, None, None], patch_i, patch_j]
    patch_mask &= gt_patch_dist <= gt_patch_radius_3d[:, None, None]

    # Pick only non-empty patches
    MINIMUM_POINTS_PER_PATCH = 32
    nonempty = torch.where(patch_mask.sum(dim=(-2, -1)) >= MINIMUM_POINTS_PER_PATCH)
    num_nonempty_patches = nonempty[0].shape[0]
    if num_nonempty_patches == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), {}
    
    # Finalize all patch variables
    patch_batch_idx, patch_i, patch_j = patch_batch_idx[nonempty], patch_i[nonempty], patch_j[nonempty]
    patch_mask = patch_mask[nonempty]                                   # [num_nonempty_patches, patch_h, patch_w]
    gt_patch_points = gt_patch_points[nonempty]                         # [num_nonempty_patches, patch_h, patch_w, 3]
    gt_patch_radius_3d = gt_patch_radius_3d[nonempty]                   # [num_nonempty_patches]
    gt_patch_anchor_points = gt_patch_anchor_points[nonempty]           # [num_nonempty_patches, 3]
    pred_patch_points = pred_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    
    # Align patch points
    (pred_patch_points_lr, gt_patch_points_lr), patch_lr_mask = mask_aware_nearest_resize((pred_patch_points, gt_patch_points), mask=patch_mask, size=(align_resolution, align_resolution))
    local_scale, local_shift = align_points_scale_xyz_shift(pred_patch_points_lr.flatten(-3, -2), gt_patch_points_lr.flatten(-3, -2), patch_lr_mask.flatten(-2) / gt_patch_radius_3d[:, None].add(1e-7), trunc=trunc)
    if global_scale is not None:
        scale_differ = local_scale / global_scale[patch_batch_idx]
        patch_valid = (scale_differ > 0.1) & (scale_differ < 10.0) & (global_scale > 0)
    else:
        patch_valid = local_scale > 0
    local_scale, local_shift = torch.where(patch_valid, local_scale, 0), torch.where(patch_valid[:, None], local_shift, 0)
    patch_mask &= patch_valid[:, None, None]

    pred_patch_points = local_scale[:, None, None, None] * pred_patch_points + local_shift[:, None, None, :]                   # [num_patches_nonempty, patch_h, patch_w, 3]
    
    # Compute loss
    gt_mean = harmonic_mean(gt_points[..., 2], gt_mask, dim=(-2, -1))
    patch_weight = patch_mask.float() / gt_patch_points[..., 2].clamp_min(0.1 * gt_mean[patch_batch_idx, None, None])          # [num_patches_nonempty, patch_h, patch_w]
    loss = _smooth((pred_patch_points - gt_patch_points).abs() * patch_weight[..., None], beta=beta).mean(dim=(-3, -2, -1))    # [num_patches_nonempty]
    
    if sparsity_aware:
        # Reweighting improves performance on sparse depth data. NOTE: this is not used in MoGe-1.
        sparsity = patch_mask.float().mean(dim=(-2, -1)) / patch_lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)
    loss = torch.scatter_reduce(torch.zeros(batch_size, dtype=dtype, device=device), dim=0, index=patch_batch_idx, src=loss, reduce='sum') / num_patches
    loss = loss.reshape(batch_shape)
    
    err = (pred_patch_points.detach() - gt_patch_points).norm(dim=-1) / gt_patch_radius_3d[..., None, None]

    # Record any scalar metric
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1), patch_mask).item(),
        'delta': weighted_mean((err < 1).float(), patch_mask).item()
    }

    return loss, misc

def normal_loss(points: torch.Tensor, gt_points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    device, dtype = points.device, points.dtype
    height, width = points.shape[-3:-1]

    leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
    upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
    leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
    downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
    rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

    gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
    gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
    gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
    gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
    gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

    mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
    mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown & (upxleft.norm(dim=-1) > 1e-6) & (gt_upxleft.norm(dim=-1) > 1e-6)
    mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup & (leftxdown.norm(dim=-1) > 1e-6) & (gt_leftxdown.norm(dim=-1) > 1e-6)
    mask_downxright = mask_leftdown & mask_rightup & mask_leftup & (downxright.norm(dim=-1) > 1e-6) & (gt_downxright.norm(dim=-1) > 1e-6)
    mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown & (rightxup.norm(dim=-1) > 1e-6) & (gt_rightxup.norm(dim=-1) > 1e-6)

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

    loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

    loss = loss.mean() / (4 * max(points.shape[-3:-1]))
    # loss = loss.sum() / (mask_upxleft.sum() + mask_leftxdown.sum() + mask_downxright.sum() + mask_rightxup.sum()).clamp(min=1)
    # loss = loss / 4 

    return loss, {}


def edge_loss(points: torch.Tensor, gt_points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    device, dtype = points.device, points.dtype
    height, width = points.shape[-3:-1]

    dx = points[..., :-1, :, :] - points[..., 1:, :, :]
    dy = points[..., :, :-1, :] - points[..., :, 1:, :]
    
    gt_dx = gt_points[..., :-1, :, :] - gt_points[..., 1:, :, :]
    gt_dy = gt_points[..., :, :-1, :] - gt_points[..., :, 1:, :]

    mask_dx = mask[..., :-1, :] & mask[..., 1:, :]
    mask_dy = mask[..., :, :-1] & mask[..., :, 1:]

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(0.1), math.radians(90), math.radians(3)

    loss_dx = mask_dx * _smooth(angle_diff_vec3(dx, gt_dx).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
    loss_dy = mask_dy * _smooth(angle_diff_vec3(dy, gt_dy).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
    loss = (loss_dx.mean(dim=(-2, -1)) + loss_dy.mean(dim=(-2, -1))) / (2 * max(points.shape[-3:-1]))

    return loss, {}


def mask_l2_loss(pred_mask: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    loss = gt_mask_neg.float() * pred_mask.square() + gt_mask_pos.float() * (1 - pred_mask).square()
    loss = loss.mean(dim=(-2, -1))
    return loss, {}


def mask_bce_loss(pred_mask_prob: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    loss = (gt_mask_pos | gt_mask_neg) * F.binary_cross_entropy(pred_mask_prob, gt_mask_pos.float(), reduction='none')
    loss = loss.mean(dim=(-2, -1))
    return loss, {}

def scale_shift_loss(pred_scale: torch.Tensor, pred_shift: torch.Tensor, gt_scale: torch.Tensor, gt_shift: torch.Tensor) -> torch.Tensor:
    loss = (torch.log(pred_scale) - torch.log(gt_scale)).square() + (pred_shift - gt_shift).square()
    loss = loss.mean()
    return loss, {}

def gram_anchoring_loss(feature_before_stabilizer: torch.Tensor, feature_after_stabilizer: torch.Tensor) -> torch.Tensor:
    feature_before_stabilizer_flatten = feature_before_stabilizer.flatten(2, 3).detach()
    feature_after_stabilizer_flatten = feature_after_stabilizer.flatten(2, 3)
    before_gram = feature_before_stabilizer_flatten.permute(0, 2, 1) @ feature_before_stabilizer_flatten
    after_gram = feature_after_stabilizer_flatten.permute(0, 2, 1) @ feature_after_stabilizer_flatten
    # normlize the gram matrix
    before_gram = before_gram / torch.linalg.matrix_norm(before_gram, ord='fro')[..., None, None]
    after_gram = after_gram / torch.linalg.matrix_norm(after_gram, ord='fro')[..., None, None]
    diff = (before_gram - after_gram).abs().mean()
    return diff, {}

def temporal_loss(pred_points: torch.Tensor, gt_points: torch.Tensor, mask: torch.Tensor, current_batch_size: int, scale_list: List[torch.Tensor], shift_list: List[torch.Tensor]) -> torch.Tensor:
    
    pred_points = pred_points.reshape(current_batch_size, -1, *pred_points.shape[1:]) # [B, S, H, W, 3]
    gt_points = gt_points.reshape(current_batch_size, -1, *gt_points.shape[1:]) # [B, S, H, W, 3]
    mask = mask.reshape(current_batch_size, -1, *mask.shape[1:]) # [B, S, H, W]
    scaled_pred_points = []
    for i in range(current_batch_size):
        scaled_pred_points.append(pred_points[i] * scale_list[i][..., None, None] + shift_list[i][..., None, :])
    scaled_pred_points = torch.stack(scaled_pred_points, dim=0)

    temporal_loss = 0
    for k in [1, 2, 4]:
        pred_diff_k = scaled_pred_points[..., :-k, :, :, :] - scaled_pred_points[..., k:, :, :, :]
        gt_diff_k = gt_points[..., :-k, :, :, :] - gt_points[..., k:, :, :, :]

        valid_diff_k = (mask[..., :-k, :, :] & mask[..., k:, :, :])

        mean_points_k = (gt_points[:, k:] + gt_points[:, :-k]) / 2

        relative_change_k = gt_diff_k.abs() / (mean_points_k.abs() + 1e-6)

        small_change_mask_k = relative_change_k.mean(dim=-1) < 0.2

        temporal_mask_k = valid_diff_k & small_change_mask_k

        if temporal_mask_k.sum() > 0:
            loss_k = F.l1_loss(pred_diff_k.sum(dim=-1)[temporal_mask_k], gt_diff_k.sum(dim=-1)[temporal_mask_k])
            temporal_loss += loss_k

    return temporal_loss, {}