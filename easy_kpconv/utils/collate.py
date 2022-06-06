from typing import List, Tuple, Dict, Callable
from itertools import chain

import numpy as np
from numpy import ndarray

from vision3d_engine.utils.tensor import array_to_tensor


def pack_points(points_list: List[ndarray]) -> Tuple[ndarray, ndarray]:
    """Pack a list of points.

    Args:
        points_list (List[array]): The list of points to pack with a length of B, each in the shape of (Ni, 3).

    Returns:
        A array of the pack points in the shape of (N', 3), where N' = sum(Ni).
        A int array of the lengths in the batch in the shape of (B).
    """
    lengths = np.asarray([points.shape[0] for points in points_list])
    points = np.concatenate(points_list, axis=0)
    return points, lengths


def collate_dict(data_dicts: List[Dict]) -> Dict:
    """Collate a batch of dict.

    The collated dict contains all keys from the batch, with each key mapped to a list of data. If a certain key is
    missing in one dict, `None` is used for padding so that all lists have the same length (the batch size).

    Args:
        data_dicts (List[Dict]): A batch of data dicts.

    Returns:
        A dict with all data collated.
    """
    keys = set(chain(*[list(data_dict.keys()) for data_dict in data_dicts]))
    collated_dict = {key: [data_dict.get(key) for data_dict in data_dicts] for key in keys}
    return collated_dict


class SimpleSingleCollateFnPackMode(Callable):
    """Simple collate function for single point cloud in pack mode.

    Note:
        1. The data of keys "points", "feats" and "labels" are packed into large tensors by stacking along axis 0. The
            names of the packed tensors are the same.
        2. A new tensor named "lengths" contains the length of each sample in the batch.
    """

    @staticmethod
    def __call__(data_dicts: List[Dict]) -> Dict:
        batch_size = len(data_dicts)

        # merge data with the same key from different samples into a list
        collated_dict = collate_dict(data_dicts)

        if batch_size == 1:
            # unwrap list if batch_size is 1
            collated_dict = {key: value[0] for key, value in collated_dict.items()}
            collated_dict["lengths"] = np.asarray([collated_dict["points"].shape[0]])
        else:
            # pack points, feats, labels and generate lengths
            points, lengths = pack_points(collated_dict.pop("points"))
            collated_dict["points"] = points
            collated_dict["lengths"] = lengths
            if "feats" in collated_dict:
                collated_dict["feats"] = np.concatenate(collated_dict.pop("feats"), axis=0)
            if "labels" in collated_dict:
                collated_dict["labels"] = np.concatenate(collated_dict.pop("labels"), axis=0)

        collated_dict["batch_size"] = batch_size

        collated_dict = array_to_tensor(collated_dict)

        return collated_dict
