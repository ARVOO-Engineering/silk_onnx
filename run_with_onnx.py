# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import os

import numpy as np
import cv2
import onnxruntime as ort

def create_img_pair_visual(
    image1,
    image2,
    img_height,
    img_width,
    matched_keypoints,
    matched_warped_keypoints,
):
    # load in images of shape (img_height, img_width)
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

    # resize if necessary
    if (img_height is not None) and (img_width is not None):
        image1 = cv2.resize(
            image1, (img_width, img_height), interpolation=cv2.INTER_AREA
        )
        image2 = cv2.resize(
            image2, (img_width, img_height), interpolation=cv2.INTER_AREA
        )

    return img_pair_visual(
        image1,
        image2,
        matched_keypoints,
        matched_warped_keypoints,
    )


def img_pair_visual(
    image1,
    image2,
    matched_keypoints,
    matched_warped_keypoints,
    good_matches_mask=None,
):
    img_width = image1.shape[1]

    height = max(image1.shape[0], image2.shape[0])

    if image1.shape[0] < height:
        image1 = np.pad(image1, ((0, height - image1.shape[0]), (0, 0), (0, 0)))

    if image2.shape[0] < height:
        image2 = np.pad(image2, ((0, height - image2.shape[0]), (0, 0), (0, 0)))

    image_pair = np.hstack((image1, image2))

    # convert keypoints to col, row (x, y) order
    matched_keypoints = matched_keypoints[:, [1, 0]]
    matched_warped_keypoints = matched_warped_keypoints[:, [1, 0]]

    matched_keypoints = matched_keypoints.astype(int)
    matched_warped_keypoints = matched_warped_keypoints.astype(int)

    # draw matched keypoint points and lines associating matched keypoints (point correspondences)
    for i in range(len(matched_keypoints)):
        img1_coords = matched_keypoints[i]
        img2_coords = matched_warped_keypoints[i]
        # add the width so the coordinates show up correctly on the second image
        img2_coords = (img2_coords[0] + img_width, img2_coords[1])

        radius = 1
        thickness = 2
        # points will be red (BGR color)
        image_pair = cv2.circle(image_pair, img1_coords, radius, (0, 0, 255), thickness)
        image_pair = cv2.circle(image_pair, img2_coords, radius, (0, 0, 255), thickness)

        thickness = 1

        if good_matches_mask is None:
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )
        else:
            if good_matches_mask[i]:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
        image_pair = cv2.line(image_pair, img1_coords, img2_coords, color, thickness)
    return image_pair


def save_image(img, output_location, output_img_name="output_img_pair.jpg"):
    file_to_save = os.path.join(output_location, output_img_name)

    # create the directory if it does not exist
    os.makedirs(output_location, exist_ok=True)

    cv2.imwrite(file_to_save, img)

# CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "../../assets/models/silk/analysis/alpha/pvgg-4.ckpt")
CHECKPOINT_PATH = "assets/models/silk/coco-rgb-aug.ckpt"
DEVICE = "cuda:0"

def match_descriptors_np(
    distances,
    max_distance=np.inf,
    cross_check=True,
    max_ratio=1.0,
):
    indices1 = np.arange(distances.shape[0])
    indices2 = np.argmin(distances, axis=1)

    if cross_check:
        matches1 = np.argmin(distances, axis=0)
        mask = indices1 == matches1[indices2]
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_distance < np.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_ratio < 1.0:
        best_distances = distances[indices1, indices2]
        distances[indices1, indices2] = np.inf
        second_best_indices2 = np.argmin(distances[indices1], axis=1)
        second_best_distances = distances[indices1, second_best_indices2]
        second_best_distances[second_best_distances == 0] = np.finfo(
            np.double
        ).eps
        ratio = best_distances / second_best_distances
        mask = ratio < max_ratio
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    matches = np.vstack((indices1, indices2))
    # matches = torch.vstack((indices1, indices2))

    return matches.T

def compute_dst_np(desc_0: np.ndarray, desc_1: np.ndarray):
    dist_type = "cosine"

    desc_0 = desc_0 / np.linalg.norm(desc_0, axis=1, keepdims=True)
    desc_1 = desc_1 / np.linalg.norm(desc_1, axis=1, keepdims=True)

    distance = 1 - np.dot(desc_0, desc_1.T)

    return distance


def mutual_nearest_neighbor(
    desc_0,
    desc_1,
    distance_fn=compute_dst_np,
    match_fn=match_descriptors_np,
    return_distances=False,
):
    dist = distance_fn(desc_0, desc_1)
    matches = match_fn(dist)
    if return_distances:
        distances = dist[(matches[:, 0], matches[:, 1])]
        return matches, distances
    return matches

def matcher(
    postprocessing="none",
    threshold=1.0,
    temperature=0.1,
    return_distances=False,
):
    
    return partial(
        mutual_nearest_neighbor,
        match_fn=partial(match_descriptors_np, max_ratio=threshold),
        distance_fn=partial(compute_dst_np),
        return_distances=return_distances,
    )
SILK_MATCHER_CPU = matcher(postprocessing="ratio-test-cpu", threshold=0.6)


def load_images(*paths):
    imagescv2 = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in paths]
    imagescv2 = [cv2.resize(image, (640, 480)) for image in imagescv2]  # resize to a common size
    imagescv2 = np.stack(imagescv2)
    # map to 0:1 range
    imagescv2 = imagescv2.astype(np.float32) / 255.0

    # images = np.stack([cv2.imread(path, cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR) for path in paths])
    # images = torch.tensor(imagescv2, device=DEVICE, dtype=torch.float32)

    # images = images.unsqueeze(1)  # add channel dimension
    images = np.expand_dims(imagescv2, axis=1)  # add channel dimension
    return images

IMAGE_0_PATH = "color1.jpg"
IMAGE_1_PATH = "color3.jpg"
OUTPUT_IMAGE_PATH = "./img.png"


def main():
    # load image
    images_0 = load_images(IMAGE_0_PATH)
    images_1 = load_images(IMAGE_1_PATH)

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
        input_feed={"images": images_0},
    )
    sparse_positions_1_onnx, sparse_descriptors_1_onnx = network.run(
        output_names=["logits", "raw_descriptors"],
        input_feed={"images": images_1},
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