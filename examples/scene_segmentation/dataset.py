from typing import Callable, List, Dict

import torch

from easy_kpconv.datasets.s3dis import S3DISBlockWiseTrainingDataset, S3DISBlockWiseTestingDataset
from easy_kpconv.datasets.utils import SceneSegTransform
from easy_kpconv.utils.collate import SimpleSingleCollateFnPackMode
from vision3d_engine.utils.dataloader import build_dataloader


def get_transform(cfg):
    train_transform = SceneSegTransform(
        use_augmentation=True,
        rotation_scale=cfg.train.augmentation_rotation_scale,
        min_scale=cfg.train.augmentation_min_scale,
        max_scale=cfg.train.augmentation_max_scale,
        noise_sigma=cfg.train.augmentation_noise_sigma,
        noise_scale=cfg.train.augmentation_noise_scale,
    )
    return train_transform


class S3DISTestingCollateFn(Callable):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.collate_fn = SimpleSingleCollateFnPackMode()

    def __call__(self, data_dicts: List[Dict]) -> Dict:
        assert len(data_dicts) == 1
        data_dict = data_dicts[0]

        batch_list = data_dict["batch_list"]
        num_batches = len(batch_list)
        new_batch_list = []
        for i in range(0, num_batches, self.batch_size):
            end_index = min(i + self.batch_size, num_batches)
            cur_data_batch = batch_list[i:end_index]
            cur_data_batch = self.collate_fn(cur_data_batch)
            if isinstance(cur_data_batch["indices"], list):
                cur_data_batch["indices"] = torch.cat(cur_data_batch["indices"], dim=0)
            new_batch_list.append(cur_data_batch)

        return {
            "batch_list": new_batch_list,
            "labels": torch.from_numpy(data_dict["labels"]),
            "inv_indices": torch.from_numpy(data_dict["inv_indices"]),
            "raw_labels": torch.from_numpy(data_dict["raw_labels"]),
        }


def train_valid_data_loader(cfg, test_area):
    train_transform = get_transform(cfg)

    train_dataset = S3DISBlockWiseTrainingDataset(
        cfg.data.dataset_dir,
        transform=train_transform,
        test_area=test_area,
        num_samples=cfg.train.num_points,
        block_size=cfg.train.block_size,
        use_normalized_location=cfg.train.use_normalized_location,
        use_z_coordinate=cfg.train.use_z_coordinate,
        point_threshold=cfg.train.point_threshold,
        pad_points=cfg.train.pad_points,
        training=True,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        collate_fn=SimpleSingleCollateFnPackMode(),
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = S3DISBlockWiseTestingDataset(
        cfg.data.dataset_dir,
        test_area=test_area,
        num_samples=cfg.test.num_points,
        block_size=cfg.test.block_size,
        block_stride=cfg.test.block_stride,
        use_normalized_location=cfg.test.use_normalized_location,
        use_z_coordinate=cfg.test.use_z_coordinate,
        pad_points=cfg.test.pad_points,
        cache_data=True,
    )

    val_loader = build_dataloader(
        val_dataset,
        batch_size=1,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=S3DISTestingCollateFn(cfg.test.batch_size),
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def test_data_loader(cfg, test_area):
    test_dataset = S3DISBlockWiseTestingDataset(
        cfg.data.dataset_dir,
        test_area=test_area,
        num_samples=cfg.test.num_points,
        block_size=cfg.test.block_size,
        block_stride=cfg.test.block_stride,
        use_normalized_location=cfg.test.use_normalized_location,
        use_z_coordinate=cfg.test.use_z_coordinate,
        pad_points=cfg.test.pad_points,
        cache_data=True,
    )

    test_loader = build_dataloader(
        test_dataset,
        batch_size=1,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=S3DISTestingCollateFn(cfg.test.batch_size),
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
