# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy

import numpy as np
import cv2
import torch

from lib.matching.mnn import estimate_homography_matched
from silk.backbones.silk.silk import SiLKVGG as SiLK
from silk.backbones.superpoint.superpoint import SuperPoint
from silk.backbones.superpoint.vgg import ParametricVGG

from silk.config.model import load_model_from_checkpoint
from silk.matching.mnn import estimate_homography, ransac_confidence
from silk.models.silk import matcher


from silk.backbones.silk.silk import from_feature_coords_to_image_coords
from silk.cli.image_pair_visualization import create_img_pair_visual, save_image
import onnxruntime as ort



# CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "../../assets/models/silk/analysis/alpha/pvgg-4.ckpt")
CHECKPOINT_PATH = "assets/models/silk/coco-rgb-aug.ckpt"
DEVICE = "cuda:0"

SILK_NMS = 0  # NMS radius, 0 = disabled
SILK_BORDER = 0  # remove detection on border, 0 = disabled
SILK_THRESHOLD = 1.0  # keypoint score thresholding, if # of keypoints is less than provided top-k, then will add keypoints to reach top-k value, 1.0 = disabled
SILK_TOP_K = 1600  # minimum number of best keypoints to output, could be higher if threshold specified above has low value
SILK_DEFAULT_OUTPUT = (  # outputs required when running the model
    "dense_positions",
    "normalized_descriptors",
    "probability",
)
SILK_SCALE_FACTOR = 1.41  # scaling of descriptor output, do not change
SILK_BACKBONE = ParametricVGG(
    use_max_pooling=False,
    padding=0,
    normalization_fn=[torch.nn.BatchNorm2d(i) for i in (64, 64, 128, 128)],
)
SILK_MATCHER = matcher(postprocessing="ratio-test", threshold=0.6)
SILK_MATCHER_CPU = matcher(postprocessing="ratio-test-cpu", threshold=0.6)
# SILK_MATCHER = matcher(postprocessing="double-softmax", threshold=0.6, temperature=0.1)
# SILK_MATCHER = matcher(postprocessing="none")


def load_model(model, checkpoint, topk, threshold):
    klass = None
    kwargs = {}
    load_args = {}

    if model.startswith("pvgg-"):
        load_args["state_dict_fn"] = lambda x: {
            k[len("_mods.model.") :]: v for k, v in x.items()
        }

        # start with defaults
        # ref : etc/backbones/silk-vgg.yaml
        klass = SiLK
        kwargs = {
            "in_channels": 1,  # grayscale
            "detection_threshold": threshold,
            "detection_top_k": topk,
            "nms_dist": 0,
            "padding": 0,
            "border_dist": 0,
            "descriptor_scale_factor": 1.41,  # sqrt(2)
            "default_outputs": ("sparse_positions", "matches"),
        }
        # default VGG backbone
        # ref : etc/backbones/silk-pvgg-defaults.yaml
        backbone_klass = ParametricVGG
        backbone_kargs = {
            "input_num_channels": 1,
            "use_max_pooling": False,
            "padding": 0,
        }
        if model == "pvgg-micro":
            # ref : etc/backbones/silk-pvgg-micro.yaml
            kwargs["lat_channels"] = 32
            kwargs["desc_channels"] = 32
            kwargs["feat_channels"] = 64

            backbone_kargs["channels"] = (64,)
            backbone_kargs["normalization_fn"] = [torch.nn.BatchNorm2d(64)]
        elif model == "pvgg-1":
            # ref : etc/backbones/silk-pvgg-1.yaml
            backbone_kargs["channels"] = (128,)
            backbone_kargs["normalization_fn"] = [torch.nn.BatchNorm2d(128)]
        elif model == "pvgg-2":
            # ref : etc/backbones/silk-pvgg-2.yaml
            backbone_kargs["channels"] = (128, 128)
            backbone_kargs["normalization_fn"] = [
                torch.nn.BatchNorm2d(128),
                torch.nn.BatchNorm2d(128),
            ]
        elif model == "pvgg-3":
            # ref : etc/backbones/silk-pvgg-3.yaml
            backbone_kargs["channels"] = (64, 128, 128)
            backbone_kargs["normalization_fn"] = [
                torch.nn.BatchNorm2d(64),
                torch.nn.BatchNorm2d(128),
                torch.nn.BatchNorm2d(128),
            ]
        elif model == "pvgg-4":
            # ref : etc/backbones/silk-pvgg-4.yaml
            backbone_kargs["channels"] = (64, 64, 128, 128)
            backbone_kargs["normalization_fn"] = [
                torch.nn.BatchNorm2d(64),
                torch.nn.BatchNorm2d(64),
                torch.nn.BatchNorm2d(128),
                torch.nn.BatchNorm2d(128),
            ]

        kwargs["backbone"] = backbone_klass(**backbone_kargs)
    elif model == "superpoint":
        load_args["state_dict_key"] = None
        load_args["map_name"] = {
            "conv1a.weight": "magicpoint.backbone._backbone.l1.0.0.weight",
            "conv1a.bias": "magicpoint.backbone._backbone.l1.0.0.bias",
            "conv1b.weight": "magicpoint.backbone._backbone.l1.1.0.weight",
            "conv1b.bias": "magicpoint.backbone._backbone.l1.1.0.bias",
            "conv2a.weight": "magicpoint.backbone._backbone.l2.0.0.weight",
            "conv2a.bias": "magicpoint.backbone._backbone.l2.0.0.bias",
            "conv2b.weight": "magicpoint.backbone._backbone.l2.1.0.weight",
            "conv2b.bias": "magicpoint.backbone._backbone.l2.1.0.bias",
            "conv3a.weight": "magicpoint.backbone._backbone.l3.0.0.weight",
            "conv3a.bias": "magicpoint.backbone._backbone.l3.0.0.bias",
            "conv3b.weight": "magicpoint.backbone._backbone.l3.1.0.weight",
            "conv3b.bias": "magicpoint.backbone._backbone.l3.1.0.bias",
            "conv4a.weight": "magicpoint.backbone._backbone.l4.0.0.weight",
            "conv4a.bias": "magicpoint.backbone._backbone.l4.0.0.bias",
            "conv4b.weight": "magicpoint.backbone._backbone.l4.1.0.weight",
            "conv4b.bias": "magicpoint.backbone._backbone.l4.1.0.bias",
            "convPa.weight": "magicpoint.backbone._heads._mods.logits._detH1.0.weight",
            "convPa.bias": "magicpoint.backbone._heads._mods.logits._detH1.0.bias",
            "convPb.weight": "magicpoint.backbone._heads._mods.logits._detH2.0.weight",
            "convPb.bias": "magicpoint.backbone._heads._mods.logits._detH2.0.bias",
            "convDa.weight": "magicpoint.backbone._heads._mods.raw_descriptors._desH1.0.weight",
            "convDa.bias": "magicpoint.backbone._heads._mods.raw_descriptors._desH1.0.bias",
            "convDb.weight": "magicpoint.backbone._heads._mods.raw_descriptors._desH2.0.weight",
            "convDb.bias": "magicpoint.backbone._heads._mods.raw_descriptors._desH2.0.bias",
        }

        # start with defaults
        # ref : etc/backbones/silk-vgg.yaml
        klass = SuperPoint
        kwargs = {
            "use_batchnorm": False,
            "default_outputs": ("positions", "sparse_descriptors"),
        }

    model = klass(**kwargs)

    model = load_model_from_checkpoint(
        model,
        checkpoint_path=checkpoint,
        device=DEVICE,
        freeze=True,
        eval=True,
        **load_args,
    )

    return model


def load_images(*paths, as_gray=True):
    imagescv2 = [cv2.imread(path, cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR) for path in paths]
    imagescv2 = [cv2.resize(image, (360, 202)) for image in imagescv2]  # resize to a common size
    imagescv2 = np.stack(imagescv2)
    # map to 0:1 range
    imagescv2 = imagescv2.astype(np.float32) / 255.0

    # images = np.stack([cv2.imread(path, cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR) for path in paths])
    images = torch.tensor(imagescv2, device=DEVICE, dtype=torch.float32)
    if not as_gray:
        images = images.permute(0, 3, 1, 2)
        images = images / 255.0
    else:
        images = images.unsqueeze(1)  # add channel dimension
    return images


def load_mask():
    # set mask to all ones except for where the bounding boxes are
    example_mask = np.ones((202, 360), dtype=np.float32)
    example_mask[0:80, 120:250] = 0.0
    example_mask = torch.tensor(example_mask, device=DEVICE, dtype=torch.float32)
    example_mask = example_mask.unsqueeze(0).unsqueeze(0)  # add batch and channel dimensions
    return example_mask


def get_model(
    checkpoint=CHECKPOINT_PATH,
    nms=SILK_NMS,
    device=DEVICE,
    default_outputs=SILK_DEFAULT_OUTPUT,
):
    # load model
    model = SiLK(
        in_channels=1,
        backbone=deepcopy(SILK_BACKBONE),
        detection_threshold=SILK_THRESHOLD,
        detection_top_k=SILK_TOP_K,
        nms_dist=nms,
        border_dist=SILK_BORDER,
        default_outputs=default_outputs,
        descriptor_scale_factor=SILK_SCALE_FACTOR,
        padding=0,
    )
    model = load_model_from_checkpoint(
        model,
        checkpoint_path=checkpoint,
        state_dict_fn=lambda x: {k[len("_mods.model.") :]: v for k, v in x.items()},
        device=device,
        freeze=True,
        eval=True,
    )
    return model

IMAGE_0_PATH = "color1.jpg"
IMAGE_1_PATH = "color3.jpg"
OUTPUT_IMAGE_PATH = "./img.png"
OUTPUT_IMAGE_PATH2 = "./img2.png"


def main():
    # load image
    images_0 = load_images(IMAGE_0_PATH, IMAGE_1_PATH)

    mask = load_mask()

    # load model
    # model = get_model(default_outputs=("sparse_positions", "sparse_descriptors"))

    model = load_model("pvgg-micro", "assets/models/silk/analysis/alpha/pvgg-micro.ckpt", SILK_TOP_K, SILK_THRESHOLD)    


    # run model
    sparse_positions, matches = model(images_0, mask)
    # get matches
    matches: np.ndarray = matches[0].detach().cpu().numpy()  # get the first batch of matches
    matches = matches[matches[:, 0] >= 0]  # filter out invalid matches
    
    # create output image
    image_pair = create_img_pair_visual(
        IMAGE_0_PATH,
        IMAGE_1_PATH,
        269,
        480,
        sparse_positions[0][matches[:, 0]].detach().cpu().numpy(),
        sparse_positions[1][matches[:, 1]].detach().cpu().numpy(),
    )

    save_image(
        image_pair,
        os.path.dirname(OUTPUT_IMAGE_PATH),
        os.path.basename(OUTPUT_IMAGE_PATH),
    )
    estimate_homography_matched(sparse_positions[0], sparse_positions[1], matches)

    model.to_onnx("model.onnx", images_0, mask, export_params=True)
    ort.set_default_logger_severity(3)  # set ONNX Runtime logger to warning level
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    network = ort.InferenceSession(
        "model.onnx", sess_options=so, providers=["CUDAExecutionProvider"]
    )

    sparse_positions_onnx, matches = network.run(
        output_names=["sparse_positions", "matches"],
        input_feed={"images": images_0.detach().cpu().numpy(), "mask": mask.detach().cpu().numpy()},
    )
    # sparse_positions_1_onnx, sparse_descriptors_1_onnx = network.run(
    #     output_names=["sparse_positions", "sparse_descriptors"],
    #     input_feed={"images": images_1.detach().cpu().numpy(), "mask": mask.detach().cpu().numpy()},
    # )
    
    matches = matches[0]
    matches = matches[matches[:, 0] >= 0]  # filter out invalid matches


    estimated_homography, mask = cv2.findHomography(
        sparse_positions_onnx[0][matches[:, 0]][:, :2],
        sparse_positions_onnx[1][matches[:, 1]][:, :2],
        cv2.RANSAC,
    )
    num_inliers = int(np.sum(mask))
    inlier_ratio = num_inliers / (len(matches) + 1e-6)

    confidence = inlier_ratio
    if estimated_homography is not None:
        dx = estimated_homography[0, 2]
        dy = estimated_homography[1, 2]
        print(
            f"Estimated homography: dx={dx:.2f}, dy={dy:.2f}, confidence={confidence:.2f}"
        )

    image_pair = create_img_pair_visual(
        IMAGE_0_PATH,
        IMAGE_1_PATH,
        480,
        640,
        sparse_positions_onnx[0][matches[:, 0]],
        sparse_positions_onnx[1][matches[:, 1]],
    )

    save_image(
        image_pair,
        os.path.dirname(OUTPUT_IMAGE_PATH2),
        os.path.basename(OUTPUT_IMAGE_PATH2),
    )



if __name__ == "__main__":
    main()