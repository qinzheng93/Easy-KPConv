import argparse

from easy_kpconv.ops.calibrate_neighbors import calibrate_neighbors_pack_mode
from easy_kpconv.utils.collate import SimpleSingleCollateFnPackMode
from easy_kpconv.datasets.s3dis import S3DISBlockWiseTrainingDataset

from config import make_cfg
from dataset import get_transform


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_area", default="Area_5", help="test area")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()

    train_transform = get_transform(cfg)
    train_dataset = S3DISBlockWiseTrainingDataset(
        cfg.data.dataset_dir,
        transform=train_transform,
        test_area=args.test_area,
        num_samples=cfg.train.num_points,
        block_size=cfg.train.block_size,
        use_normalized_location=cfg.train.use_normalized_location,
        use_z_coordinate=cfg.train.use_z_coordinate,
        point_threshold=cfg.train.point_threshold,
        pad_points=cfg.train.pad_points,
        training=True,
    )

    neighbor_limits = calibrate_neighbors_pack_mode(
        train_dataset,
        SimpleSingleCollateFnPackMode,
        cfg.model.num_stages,
        cfg.model.basic_voxel_size,
        cfg.model.basic_voxel_size * cfg.model.kpconv_radius,
    )

    print(f"Neighbor limits: {neighbor_limits}. Please fill neighbor_limits in config.py.")


if __name__ == "__main__":
    main()
