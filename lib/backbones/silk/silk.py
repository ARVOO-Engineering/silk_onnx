# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Iterable, Tuple, Union

import torch
import torch.nn as nn

from silk.backbones.abstract.shared_backbone_multiple_heads import (
    SharedBackboneMultipleHeads,
)
from silk.backbones.loftr.resnet_fpn import ResNetFPN_8_2
from silk.backbones.superpoint.magicpoint import (
    Backbone as VGGBackbone,
    DetectorHead as VGGDetectorHead,
    MagicPoint,
)
from silk.backbones.superpoint.superpoint import (
    DescriptorHead as VGGDescriptorHead,
    SuperPoint,
)
from silk.flow import AutoForward, Flow
from silk.models.superpoint_utils import get_dense_positions
import torch.nn.functional as F



def from_feature_coords_to_image_coords(model, desc_positions):
    if isinstance(desc_positions, tuple):
        return tuple(
            from_feature_coords_to_image_coords(
                model,
                dp,
            )
            for dp in desc_positions
        )
    coord_mapping = model.coordinate_mapping_composer.get("images", "raw_descriptors")
    desc_positions = torch.cat(
        [
            coord_mapping.reverse(desc_positions[..., :2]),
            desc_positions[..., 2:],
        ],
        dim=-1,
    )

    return desc_positions


class SiLKBase(AutoForward, torch.nn.Module):
    def __init__(
        self,
        backbone,
        input_name: str = "images",
        mask_name: str = "mask",
        backbone_output_name: Union[str, Tuple[str]] = "features",
        default_outputs: Union[str, Iterable[str]] = ("descriptors", "score"),
    ):
        torch.nn.Module.__init__(self)

        self.backbone = SharedBackboneMultipleHeads(
            backbone=backbone,
            input_name=input_name,
            mask_name=mask_name,
            backbone_output_name=backbone_output_name,
        )

        self.detector_heads = set()
        self.descriptor_heads = set()

        AutoForward.__init__(self, self.backbone.flow, default_outputs=default_outputs)

    @property
    def coordinate_mapping_composer(self):
        return self.backbone.coordinate_mapping_composer

    def add_detector_head(self, head_name, head, backbone_output_name=None):
        self.backbone.add_head_to_backbone_output(head_name, head, backbone_output_name)
        self.detector_heads.add(head_name)

    def add_descriptor_head(self, head_name, head, backbone_output_name=None):
        self.backbone.add_head_to_backbone_output(head_name, head, backbone_output_name)
        self.descriptor_heads.add(head_name)


class SiLKVGG(SiLKBase):
    def __init__(
        self,
        in_channels,
        *,
        feat_channels: int = 128,
        lat_channels: int = 128,
        desc_channels: int = 128,
        use_batchnorm: bool = True,
        backbone=None,
        detector_head=None,
        descriptor_head=None,
        detection_threshold: float = 0.8,
        detection_top_k: int = 100,
        nms_dist=4,
        border_dist=4,
        descriptor_scale_factor: float = 1.0,
        learnable_descriptor_scale_factor: bool = False,
        normalize_descriptors: bool = True,
        padding: int = 1,
        **base_kwargs,
    ) -> None:
        backbone = (
            VGGBackbone(
                num_channels=in_channels,
                use_batchnorm=use_batchnorm,
                use_max_pooling=False,
                padding=padding,
            )
            if backbone is None
            else backbone
        )

        detector_head = (
            VGGDetectorHead(
                in_channels=feat_channels,
                lat_channels=lat_channels,
                out_channels=1,
                use_batchnorm=use_batchnorm,
                padding=padding,
            )
            if detector_head is None
            else detector_head
        )

        descriptor_head = (
            VGGDescriptorHead(
                in_channels=feat_channels,
                out_channels=desc_channels,
                use_batchnorm=use_batchnorm,
                padding=padding,
            )
            if descriptor_head is None
            else descriptor_head
        )

        SiLKBase.__init__(
            self,
            backbone=backbone,
            **base_kwargs,
        )

        self.add_detector_head("logits", detector_head)
        self.add_descriptor_head("raw_descriptors", descriptor_head)

        self.descriptor_scale_factor = nn.parameter.Parameter(
            torch.tensor(descriptor_scale_factor),
            requires_grad=learnable_descriptor_scale_factor,
        )
        self.normalize_descriptors = normalize_descriptors

        MagicPoint.add_detector_head_post_processing(
            self.flow,
            "logits",
            prefix="",
            cell_size=1,
            detection_threshold=detection_threshold,
            detection_top_k=detection_top_k,
            nms_dist=nms_dist,
            border_dist=border_dist,
        )

        SiLKVGG.add_descriptor_head_post_processing(
            self.flow,
            input_name=self.backbone.input_name,
            descriptor_head_output_name="raw_descriptors",
            prefix="",
            scale_factor=self.descriptor_scale_factor,
            normalize_descriptors=normalize_descriptors,
        )

        SiLKVGG.add_matcher_post_processing(
            self.flow,
            matcher_output_name="matches",
        )

    def to_onnx(self, file_path: str, dummy_input: torch.Tensor, dummy_mask_input, export_params: bool = True):
        import torch.onnx
        import onnx
        import onnxsim

        torch.onnx.export(
            self,
            (dummy_input, dummy_mask_input),
            file_path,
            export_params=export_params,
            verbose=False,
            opset_version=13,
            do_constant_folding=True,
            input_names=["images", "mask"],
            output_names=["sparse_positions", "matches"],
        )
        onnx_model = onnx.load(file_path)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        onnx_model, _ = onnxsim.simplify(onnx_model)  # simplify model
        onnx.save(onnx_model, file_path)  # save simplified model

    @staticmethod
    def add_descriptor_head_post_processing(
        flow: Flow,
        input_name: str = "images",
        descriptor_head_output_name: str = "raw_descriptors",
        positions_name: str = "positions",
        prefix: str = "superpoint.",
        scale_factor: float = 1.0,
        normalize_descriptors: bool = True,
    ):
        flow.define_transition(
            f"{prefix}normalized_descriptors",
            partial(
                SuperPoint.normalize_descriptors,
                scale_factor=scale_factor,
                normalize=normalize_descriptors,
            ),
            descriptor_head_output_name,
        )
        flow.define_transition(
            f"{prefix}dense_descriptors",
            SiLKVGG.get_dense_descriptors,
            f"{prefix}normalized_descriptors",
        )
        flow.define_transition(f"{prefix}image_size", SuperPoint.image_size, input_name)
        flow.define_transition(
            f"{prefix}sparse_descriptors",
            partial(
                SiLKVGG.sparsify_descriptors,
                scale_factor=scale_factor,
                normalize_descriptors=normalize_descriptors,
            ),
            descriptor_head_output_name,
            positions_name,
        )
        flow.define_transition(
            f"{prefix}sparse_positions",
            lambda x: x,
            positions_name,
        )
        flow.define_transition(
            f"{prefix}dense_positions",
            SiLKVGG.get_dense_positions,
            "probability",
        )

    @staticmethod
    def add_matcher_post_processing(
        flow: Flow,
        matcher_output_name: str = "matches",
    ):
        flow.define_transition(
            'distance',
            SiLKVGG.compute_dist,
            "sparse_descriptors",
        )

        flow.define_transition(
            matcher_output_name,
            SiLKVGG.match_descriptors,
            'distance',
        )
            
        
    
    # def mutual_nearest_neighbor(
    #     desc_0,
    #     desc_1,
    # ):
    #     dist = distance_fn(desc_0, desc_1)
    #     matches = match_fn(dist)
    #     return matches

    @staticmethod
    def match_descriptors(
        distances,
        max_distance=torch.inf,
        cross_check=True,
        max_ratio=0.7,
    ):
            # Step 1: Prepare index1 (row indices)
        indices1 = torch.arange(distances.shape[0], device=distances.device)

        # Step 2: Get best match (argmin)
        indices2 = torch.argmin(distances, dim=1)

        # Step 3: Gather best distances
        best_distances = torch.gather(distances, 1, indices2.unsqueeze(1)).squeeze(1)

        # Step 4: Mask out best match by setting it to inf
        distances_masked = distances.clone()
        distances_masked.scatter_(1, indices2.unsqueeze(1), float("inf"))

        # Step 5: Find second-best match
        second_best_indices2 = torch.argmin(distances_masked, dim=1)
        second_best_distances = torch.gather(distances_masked, 1, second_best_indices2.unsqueeze(1)).squeeze(1)

        # Step 6: Avoid divide-by-zero
        epsilon = torch.finfo(best_distances.dtype).eps
        safe_second_best = torch.where(second_best_distances == 0, torch.tensor(epsilon, device=distances.device), second_best_distances)

        # Step 7: Compute Lowe's ratio
        ratio = best_distances / safe_second_best
        mask_ratio = ratio < max_ratio

        # Step 8: Apply Lowe's ratio filter
        invalid = torch.full_like(indices1, -1)
        indices1 = torch.where(mask_ratio, indices1, invalid)
        indices2 = torch.where(mask_ratio, indices2, invalid)

        # Step 9: Cross-check filter
        if cross_check:
            # Find best match from descriptors2 back to descriptors1
            matches1 = torch.argmin(distances, dim=0)  # [N2] => indices in descriptors1

            # matches1[indices2] should equal indices1 (only valid if not -1)
            valid_check = (indices2 >= 0) & (matches1[indices2] == indices1)
            indices1 = torch.where(valid_check, indices1, invalid)
            indices2 = torch.where(valid_check, indices2, invalid)

        matches = torch.stack((indices1, indices2), dim=1)
        return (matches, matches)
    @staticmethod        
    def compute_dist(descriptors):
        ''' Compute distance between two sets of descriptors.'''
        assert len(descriptors) == 2, "Descriptors should be a tuple of two sets of descriptors."
        # assert dist_type in {"dot", "cosine", "l2"}

        # if dist_type == "dot":
        #     distance = 1 - torch.matmul(desc_0, desc_1.T)
        # elif dist_type == "cosine":
        desc_0 = F.normalize(
            descriptors[0],
            p=2,
            dim=1,
        )
        desc_1 = F.normalize(
            descriptors[1],
            p=2,
            dim=1,
        )
        distance = 1 - torch.matmul(desc_0, desc_1.T)
        # elif dist_type == "l2":
        #     distance = torch.cdist(desc_0, desc_1, p=2)

        return distance


    @staticmethod
    def get_dense_positions(probability):
        batch_size = probability.shape[0]
        device = probability.device
        dense_positions = get_dense_positions(
            probability.shape[2],
            probability.shape[3],
            device,
            batch_size=batch_size,
        )

        dense_probability = probability.reshape(probability.shape[0], -1, 1)
        dense_positions = torch.cat((dense_positions, dense_probability), dim=2)

        return dense_positions

    @staticmethod
    def get_dense_descriptors(normalized_descriptors):
        dense_descriptors = normalized_descriptors.reshape(
            normalized_descriptors.shape[0],
            normalized_descriptors.shape[1],
            -1,
        )
        dense_descriptors = dense_descriptors.permute(0, 2, 1)
        return dense_descriptors

    @staticmethod
    def sparsify_descriptors(
        raw_descriptors,
        positions,
        scale_factor: float = 1.0,
        normalize_descriptors: bool = True,
    ):
        sparse_descriptors = []
        for i, pos in enumerate(positions):
            pos = pos[:, :2]
            pos = pos.floor().long()

            descriptors = raw_descriptors[i, :, pos[:, 0], pos[:, 1]].T

            # L2 normalize the descriptors
            descriptors = SuperPoint.normalize_descriptors(
                descriptors,
                scale_factor,
                normalize_descriptors,
            )

            sparse_descriptors.append(descriptors)
        return tuple(sparse_descriptors)

class SiLKLoFTR(SiLKBase):
    def __init__(
        self,
        in_channels,
        *,
        initial_dim: int = 128,
        block_dims: Tuple[int] = (128, 196, 256),
        lat_channels: int = 256,
        desc_channels: int = 256,
        use_batchnorm: bool = True,
        backbone=None,
        detector_head=None,
        descriptor_head=None,
        detection_threshold: float = 0.8,
        detection_top_k: int = 100,
        nms_dist=4,
        border_dist=4,
        descriptor_scale_factor: float = 1.0,
        learnable_descriptor_scale_factor: bool = False,
        resolution_preserving: bool = False,
        padding: int = 1,
        **base_kwargs,
    ) -> None:
        backbone = (
            ResNetFPN_8_2(
                {
                    "in_channels": in_channels,
                    "initial_dim": initial_dim,
                    "block_dims": block_dims,
                    "resolution_preserving": resolution_preserving,
                    "padding": padding,
                }
            )
            if backbone is None
            else backbone
        )

        feat_channels = block_dims[0]

        detector_head = (
            VGGDetectorHead(
                in_channels=feat_channels,
                lat_channels=lat_channels,
                out_channels=1,
                use_batchnorm=use_batchnorm,
                padding=padding,
            )
            if detector_head is None
            else detector_head
        )

        descriptor_head = (
            VGGDescriptorHead(
                in_channels=feat_channels,
                out_channels=desc_channels,
                use_batchnorm=use_batchnorm,
                padding=padding,
            )
            if descriptor_head is None
            else descriptor_head
        )

        super().__init__(
            backbone=backbone,
            backbone_output_name=("low_res_features", "features"),
            **base_kwargs,
        )

        self.add_detector_head("logits", detector_head, backbone_output_name="features")
        self.add_descriptor_head(
            "raw_descriptors",
            descriptor_head,
            backbone_output_name="features",
        )

        # TODO : Learneable ?
        if learnable_descriptor_scale_factor:
            self.descriptor_scale_factor = nn.parameter.Parameter(
                torch.tensor(descriptor_scale_factor)
            )
        else:
            self.descriptor_scale_factor = descriptor_scale_factor

        MagicPoint.add_detector_head_post_processing(
            self.flow,
            "logits",
            prefix="",
            cell_size=1,
            detection_threshold=detection_threshold,
            detection_top_k=detection_top_k,
            nms_dist=nms_dist,
            border_dist=border_dist,
        )

        SiLKVGG.add_descriptor_head_post_processing(
            self.flow,
            input_name=self.backbone.input_name,
            descriptor_head_output_name="raw_descriptors",
            prefix="",
            scale_factor=self.descriptor_scale_factor,
        )
