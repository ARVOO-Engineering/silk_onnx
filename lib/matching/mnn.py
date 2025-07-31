# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from silk.logger import LOG


def compute_dist(desc_0, desc_1, dist_type="dot"):
    assert dist_type in {"dot", "cosine", "l2"}

    if dist_type == "dot":
        distance = 1 - torch.matmul(desc_0, desc_1.T)
    elif dist_type == "cosine":
        desc_0 = F.normalize(
            desc_0,
            p=2,
            dim=1,
        )
        desc_1 = F.normalize(
            desc_1,
            p=2,
            dim=1,
        )
        distance = 1 - torch.matmul(desc_0, desc_1.T)
    elif dist_type == "l2":
        distance = torch.cdist(desc_0, desc_1, p=2)

    return distance

def compute_dst_np(desc_0: np.ndarray, desc_1: np.ndarray):
    dist_type = "cosine"

    desc_0 = desc_0 / np.linalg.norm(desc_0, axis=1, keepdims=True)
    desc_1 = desc_1 / np.linalg.norm(desc_1, axis=1, keepdims=True)

    distance = 1 - np.dot(desc_0, desc_1.T)

    return distance


def double_softmax_distance(desc_0, desc_1, temperature=1.0):
    similarity = torch.matmul(desc_0, desc_1.T) / temperature
    matching_probability = torch.softmax(similarity, dim=0) * torch.softmax(
        similarity, dim=1
    )
    return 1.0 - matching_probability


def match_descriptors(
    distances,
    max_distance=torch.inf,
    cross_check=True,
    max_ratio=1.0,
):
    indices1 = torch.arange(distances.shape[0], device=distances.device)
    indices2 = torch.argmin(distances, dim=1)

    if cross_check:
        matches1 = torch.argmin(distances, dim=0)
        mask = indices1 == matches1[indices2]
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_distance < torch.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_ratio < 1.0:
        best_distances = distances[indices1, indices2]
        distances[indices1, indices2] = torch.inf
        second_best_indices2 = torch.argmin(distances[indices1], axis=1)
        second_best_distances = distances[indices1, second_best_indices2]
        second_best_distances[second_best_distances == 0] = torch.finfo(
            torch.double
        ).eps
        ratio = best_distances / second_best_distances
        mask = ratio < max_ratio
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    matches = torch.vstack((indices1, indices2))

    return matches.T



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



def swap_xy(given_ordering, required_ordering, positions):
    assert given_ordering in {"yx", "xy"}
    assert required_ordering in {"yx", "xy"}

    if given_ordering == required_ordering:
        return positions

    return positions[..., [1, 0]]


def mutual_nearest_neighbor(
    desc_0,
    desc_1,
    distance_fn=compute_dist,
    match_fn=match_descriptors,
    return_distances=False,
):
    dist = distance_fn(desc_0, desc_1)
    matches = match_fn(dist)
    if return_distances:
        distances = dist[(matches[:, 0], matches[:, 1])]
        return matches, distances
    return matches


def ransac(matched_points_0, matched_points_1, ordering="xy"):
    assert len(matched_points_0) == len(matched_points_1)

    if len(matched_points_0) < 4:
        LOG.warning(
            f"ransac cannot be run, only {len(matched_points_0)} were provided (<4)"
        )
        return None

    matched_points_0 = swap_xy(ordering, "xy", matched_points_0)
    matched_points_1 = swap_xy(ordering, "xy", matched_points_1)

    matched_points_0 = matched_points_0.detach().cpu().numpy()
    matched_points_1 = matched_points_1.detach().cpu().numpy()

    estimated_homography, _ = cv2.findHomography(
        matched_points_0,
        matched_points_1,
        cv2.RANSAC,
    )

    if estimated_homography is not None:
        estimated_homography = torch.tensor(
            estimated_homography,
            dtype=torch.float32,
        )

    return estimated_homography

def ransac_confidence(matched_points_0, matched_points_1, ordering="xy"):
    assert len(matched_points_0) == len(matched_points_1)

    if len(matched_points_0) < 4:
        LOG.warning(
            f"ransac cannot be run, only {len(matched_points_0)} were provided (<4)"
        )
        return None

    matched_points_0 = swap_xy(ordering, "xy", matched_points_0)
    matched_points_1 = swap_xy(ordering, "xy", matched_points_1)

    matched_points_0 = matched_points_0.detach().cpu().numpy()
    matched_points_1 = matched_points_1.detach().cpu().numpy()

    estimated_homography, mask = cv2.findHomography(
        matched_points_0,
        matched_points_1,
        cv2.RANSAC,
    )

    
    num_inliers = int(np.sum(mask))
    inlier_ratio = num_inliers / (len(matched_points_0) + 1e-6)

    confidence = inlier_ratio
    if estimated_homography is not None:
        dx = estimated_homography[0, 2]
        dy = estimated_homography[1, 2]
        print(
            f"Estimated homography: dx={dx:.2f}, dy={dy:.2f}, confidence={confidence:.2f}"
        )

    if estimated_homography is not None:
        estimated_homography = torch.tensor(
            estimated_homography,
            dtype=torch.float32,
        )

    return estimated_homography


def batched_ransac(matched_points_0, matched_points_1, ordering="xy"):
    return [
        ransac(mp0, mp1, ordering)
        for mp0, mp1 in zip(matched_points_0, matched_points_1)
    ]


def estimate_homography(
    points_0,
    points_1,
    desc_0,
    desc_1,
    matcher_fn=mutual_nearest_neighbor,
    homography_solver_fn=ransac_confidence,
    ordering="xy",
):
    assert ordering in {"xy", "yx"}

    matches = matcher_fn(desc_0, desc_1)
    matched_points_0 = points_0[matches[:, 0]]
    matched_points_1 = points_1[matches[:, 1]]

    estimated_homography = homography_solver_fn(
        matched_points_0[:, :2],
        matched_points_1[:, :2],
        ordering,
    )

    return (
        estimated_homography,
        matched_points_0,
        matched_points_1,
    )


def batched_estimate_homography(
    points_0,
    points_1,
    desc_0,
    desc_1,
    matcher_fn=mutual_nearest_neighbor,
    homography_solver_fn=ransac,
    ordering="xy",
):
    estimated_homography = []
    matched_points_0 = []
    matched_points_1 = []

    for args in zip(
        points_0,
        points_1,
        desc_0,
        desc_1,
    ):
        result = estimate_homography(
            *args,
            matcher_fn=matcher_fn,
            homography_solver_fn=homography_solver_fn,
            ordering=ordering,
        )
        estimated_homography.append(result[0])
        matched_points_0.append(result[1])
        matched_points_1.append(result[2])

    return estimated_homography, matched_points_0, matched_points_1
