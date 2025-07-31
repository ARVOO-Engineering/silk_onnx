# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy

import numpy as np
import cv2
import torch

from silk.backbones.silk.silk import SiLKVGG as SiLK
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
SILK_TOP_K = 1000  # minimum number of best keypoints to output, could be higher if threshold specified above has low value
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


def load_images(*paths, as_gray=True):
    imagescv2 = [cv2.imread(path, cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR) for path in paths]
    imagescv2 = [cv2.resize(image, (640, 480)) for image in imagescv2]  # resize to a common size
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


def model_with_corrected_positions(model):
    def fn(images):
        results = model(images)
        assert type(results) is tuple
        positions = results[
            0
        ]  # IMPORTANT : only works when positions are in first place
        positions = from_feature_coords_to_image_coords(model, positions)
        return (positions,) + results[1:]
    model.__call__ = fn

    return model

IMAGE_0_PATH = "color1.jpg"
IMAGE_1_PATH = "color3.jpg"
OUTPUT_IMAGE_PATH = "./img.png"


def main():
    # load image
    images_0 = load_images(IMAGE_0_PATH)
    images_1 = load_images(IMAGE_1_PATH)

    # load model
    model = get_model(default_outputs=("sparse_positions", "sparse_descriptors"))

    # run model
    sparse_positions_0, sparse_descriptors_0 = model(images_0)
    sparse_positions_1, sparse_descriptors_1 = model(images_1)

    sparse_positions_0 = from_feature_coords_to_image_coords(model, sparse_positions_0)
    sparse_positions_1 = from_feature_coords_to_image_coords(model, sparse_positions_1)

    # get matches
    matches = SILK_MATCHER(sparse_descriptors_0[0], sparse_descriptors_1[0])
    estimate_homography(sparse_positions_0[0], sparse_positions_1[0], sparse_descriptors_0[0], sparse_descriptors_1[0])

    # create output image
    image_pair = create_img_pair_visual(
        IMAGE_0_PATH,
        IMAGE_1_PATH,
        480,
        640,
        sparse_positions_0[0][matches[:, 0]].detach().cpu().numpy(),
        sparse_positions_1[0][matches[:, 1]].detach().cpu().numpy(),
    )

    save_image(
        image_pair,
        os.path.dirname(OUTPUT_IMAGE_PATH),
        os.path.basename(OUTPUT_IMAGE_PATH),
    )

    model = model_with_corrected_positions(model)

    print(f"result saved in {OUTPUT_IMAGE_PATH}")
    print("done")
    model.to_onnx("model.onnx", images_0, export_params=True)
    ort.set_default_logger_severity(3)  # set ONNX Runtime logger to warning level
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    network = ort.InferenceSession(
        "model.onnx", sess_options=so, providers=["CPUExecutionProvider"]
    )

    # height, width = network.get_inputs()[0].shape[2:]
    # dtype = np.float16 if "float16" in network.get_inputs()[0].type else np.float32
    # batch_size = 1  # set batch size for inference
    # width_ratio: float = 640 / width
    # height_ratio: float = 480 / height
    # ratio = None
    # if width_ratio == height_ratio:
    #     ratio = width_ratio
    sparse_positions_0_onnx, sparse_descriptors_0_onnx = network.run(
        output_names=["logits", "raw_descriptors"],
        input_feed={"images": images_0.detach().cpu().numpy()},
    )
    sparse_positions_1_onnx, sparse_descriptors_1_onnx = network.run(
        output_names=["logits", "raw_descriptors"],
        input_feed={"images": images_1.detach().cpu().numpy()},
    )
    matches = SILK_MATCHER_CPU(sparse_descriptors_0_onnx, sparse_descriptors_1_onnx)


    estimated_homography, mask = cv2.findHomography(
        sparse_positions_0_onnx[matches[:, 0]][:, :2],
        sparse_positions_1_onnx[matches[:, 1]][:, :2],
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
        sparse_positions_0_onnx[matches[:, 0]],
        sparse_positions_1_onnx[matches[:, 1]],
    )

    save_image(
        image_pair,
        os.path.dirname(OUTPUT_IMAGE_PATH),
        os.path.basename(OUTPUT_IMAGE_PATH),
    )



if __name__ == "__main__":
    main()