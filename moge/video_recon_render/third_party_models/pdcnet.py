from typing import *
from pathlib import Path
import pickle

import torch

from .DenseMatching.models.PDCNet.PDCNet import PDCNetModel, PDCNet_vgg16


def failed_to_find_class(module_, name_):
    class FailedToFindClass:
        module = module_
        name = name_
        def __repr__(self) -> str:
            return f"FailedToFindClass(module={self.module}, name={self.name})."
        
    return FailedToFindClass


class IgnoreUnknownClassUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super(IgnoreUnknownClassUnpickler, self).find_class(module, name)
        except ImportError:
            return failed_to_find_class(module, name)


class IgnoreUnknownClassPickleModule:
    Unpickler = IgnoreUnknownClassUnpickler


def load_model(checkpoint_path: Union[str, Path] = "ckpts/PDCNet_megadepth.pth.tar", device: Union[str, torch.device] = 'cuda'):
    global_optim_iter = 3
    local_optim_iter = 3
    global_gocor_arguments = {'optim_iter': global_optim_iter, 'steplength_reg': 0.1, 'train_label_map': False,
                                'apply_query_loss': True,
                                'reg_kernel_size': 3, 'reg_inter_dim': 16, 'reg_output_dim': 16}
    local_gocor_arguments = {'optim_iter': local_optim_iter, 'steplength_reg': 0.1}

    pdcnet_model = PDCNet_vgg16(
        global_corr_type='GlobalGOCor', global_gocor_arguments=global_gocor_arguments,
        normalize='leakyrelu', same_local_corr_at_all_levels=True,
        local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
        local_decoder_type='OpticalFlowEstimatorResidualConnection',
        global_decoder_type='CMDTopResidualConnection',
        corr_for_corr_uncertainty_decoder='corr',
        give_layer_before_flow_to_uncertainty_decoder=True,
        var_2_plus=520 ** 2, var_2_plus_256=256 ** 2, var_1_minus_plus=1.0, var_2_minus=2.0
    )
    
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', pickle_module=IgnoreUnknownClassPickleModule)

    if 'state_dict' in checkpoint_dict:
        checkpoint_dict = checkpoint_dict['state_dict']

    pdcnet_model.load_state_dict(checkpoint_dict, strict=True)

    pdcnet_model.set_inference_parameters(
        confidence_R=1.0,
        ransac_thresh=1.0,
        multi_stage_type='MS',
        mask_type_for_2_stage_alignment='proba_interval_1_above_10',
        homography_visibility_mask=False,
        list_resizing_ratios=[0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
        compute_cyclic_consistency_error=False
    )

    pdcnet_model = pdcnet_model.to(device)

    return pdcnet_model


@torch.no_grad()
def predict_flow(pdcnet_model: PDCNetModel, query_image: torch.Tensor, reference_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.is_floating_point(query_image):
        query_image = (query_image.clamp(0, 1) * 255).to(torch.uint8)
    if torch.is_floating_point(reference_image):
        reference_image = (reference_image.clamp(0, 1) * 255).to(torch.uint8)

    # PDCNet inputs require batch dimension [1, C, H, W], and it only supports batch size 1 due to a bug (maybe) in their code.
    estimated_flow, uncertainty_components = pdcnet_model.estimate_flow_and_confidence_map(query_image[None], reference_image[None], mode='channel_first')
    estimated_flow = estimated_flow[0].permute(1, 2, 0) / torch.tensor([query_image.shape[-1], query_image.shape[-2]], dtype=estimated_flow.dtype, device=estimated_flow.device)
    confidence_map = uncertainty_components['p_r'][0, 0]

    return estimated_flow, confidence_map