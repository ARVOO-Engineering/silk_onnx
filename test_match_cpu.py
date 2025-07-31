
from functools import partial

import time

import numpy as np
start_time = time.time()


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
print("--- %s seconds ---" % (time.time() - start_time))

sparse_descriptors_0 = np.random.rand(1000, 128).astype(np.float32)
sparse_descriptors_1 = np.random.rand(1000, 128).astype(np.float32)


matches = matcher(sparse_descriptors_0, sparse_descriptors_1)
print("--- %s seconds ---" % (time.time() - start_time))
