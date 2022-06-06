import random
from typing import Optional

import numpy as np
from numpy import ndarray


# Basic transforms


def normalize_points(points: ndarray) -> ndarray:
    """Normalize point cloud to a unit sphere at origin."""
    points = points - points.mean(axis=0)
    points = points / np.max(np.linalg.norm(points, axis=1))
    return points


def normalize_points_on_xy_plane(points: ndarray) -> ndarray:
    """Normalize point cloud along x-y plane in place."""
    barycenter_2d = np.mean(points[:, :2], axis=0)
    points[:, :2] -= barycenter_2d
    return points


def sample_points(points: ndarray, num_samples: int, normals: Optional[ndarray] = None):
    """Sample the first K points."""
    points = points[:num_samples]
    if normals is None:
        return points
    normals = normals[:num_samples]
    return points, normals


def random_sample_points(points: ndarray, num_samples: int, normals: Optional[ndarray] = None):
    """Randomly sample points."""
    num_points = points.shape[0]
    sel_indices = np.random.permutation(num_points)
    if num_points > num_samples:
        sel_indices = sel_indices[:num_samples]
    elif num_points < num_samples:
        num_iterations = num_samples // num_points
        num_paddings = num_samples % num_points
        all_sel_indices = [sel_indices for _ in range(num_iterations)]
        if num_paddings > 0:
            all_sel_indices.append(sel_indices[:num_paddings])
        sel_indices = np.concatenate(all_sel_indices, axis=0)
    points = points[sel_indices]
    if normals is None:
        return points
    normals = normals[sel_indices]
    return points, normals


def random_scale_points(points: ndarray, low: float = 0.8, high: float = 1.2) -> ndarray:
    """Randomly rescale point cloud."""
    scale = random.uniform(low, high)
    points = points * scale
    return points


def random_shift_points(points: ndarray, shift: float = 0.2) -> ndarray:
    bias = np.random.uniform(low=-shift, high=shift, size=(1, 3))
    points = points + bias
    return points


def random_scale_shift_points(
    points: ndarray,
    low: float = 2.0 / 3.0,
    high: float = 3.0 / 2.0,
    shift: float = 0.2,
    normals: Optional[ndarray] = None,
):
    """Randomly sigma and shift point cloud."""
    scale = np.random.uniform(low=low, high=high, size=(1, 3))
    bias = np.random.uniform(low=-shift, high=shift, size=(1, 3))
    points = points * scale + bias
    if normals is None:
        return points
    normals = normals * scale
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def random_rotate_points_along_up_axis(
    points: ndarray,
    rotation_scale: float = 1.0,
    normals: Optional[ndarray] = None,
):
    """Randomly rotate point cloud along z-axis."""
    theta = np.random.rand() * 2.0 * np.pi * rotation_scale
    # fmt: off
    rotation_t = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    # fmt: on
    points = np.matmul(points, rotation_t)
    if normals is None:
        return points
    normals = np.matmul(normals, rotation_t)
    return points, normals


def random_jitter_points(points: ndarray, sigma: float = 0.01, scale: float = 0.05) -> ndarray:
    """Randomly jitter point cloud."""
    noises = np.clip(np.random.normal(scale=sigma, size=points.shape), a_min=-scale, a_max=scale)
    points = points + noises
    return points


def random_shuffle_points(points: ndarray, normals: Optional[ndarray] = None):
    """Randomly permute point cloud."""
    indices = np.random.permutation(points.shape[0])
    points = points[indices]
    if normals is None:
        return points
    normals = normals[indices]
    return points, normals


def random_dropout_points(points: ndarray, max_p: float) -> ndarray:
    """Randomly dropout point cloud proposed in PointNet++."""
    num_points = points.shape[0]
    p = np.random.rand(num_points) * max_p
    masks = np.random.rand(num_points) < p
    points[masks] = points[0]
    return points
