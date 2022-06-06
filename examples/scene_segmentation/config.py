import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from vision3d_engine.utils.io import ensure_dir
import easy_kpconv.datasets.s3dis as s3dis

_C = edict()

# exp
_C.exp = edict()
_C.exp.name = osp.basename(osp.dirname(osp.realpath(__file__)))
_C.exp.working_dir = osp.dirname(osp.realpath(__file__))
_C.exp.output_dir = osp.join(_C.exp.working_dir, "outputs")
_C.exp.checkpoint_dir = osp.join(_C.exp.output_dir, "checkpoints")
_C.exp.log_dir = osp.join(_C.exp.output_dir, "logs")
_C.exp.event_dir = osp.join(_C.exp.output_dir, "events")
_C.exp.seed = 7351

ensure_dir(_C.exp.output_dir)
ensure_dir(_C.exp.checkpoint_dir)
ensure_dir(_C.exp.log_dir)
ensure_dir(_C.exp.event_dir)

# data
_C.data = edict()
_C.data.dataset_dir = "/data/S3DIS"
_C.data.num_classes = s3dis.NUM_CLASSES
_C.data.class_names = s3dis.CLASS_NAMES

# train
_C.train = edict()
_C.train.batch_size = 16
_C.train.num_workers = 8
_C.train.num_points = None
_C.train.point_threshold = 1024
_C.train.block_size = 2.5
_C.train.use_normalized_location = False
_C.train.use_z_coordinate = True
_C.train.pad_points = False
_C.train.augmentation_rotation_scale = 1.0
_C.train.augmentation_min_scale = 0.9
_C.train.augmentation_max_scale = 1.1
_C.train.augmentation_noise_sigma = 0.01
_C.train.augmentation_noise_scale = 0.05

# test
_C.test = edict()
_C.test.batch_size = 8
_C.test.num_workers = 0
_C.test.num_points = None
_C.test.point_threshold = 2048
_C.test.block_size = 2.5
_C.test.block_stride = 1.0
_C.test.use_normalized_location = False
_C.test.use_z_coordinate = True
_C.test.pad_points = False

# trainer
_C.trainer = edict()
_C.trainer.max_epoch = 300
_C.trainer.num_iters_per_epoch = 3000
_C.trainer.grad_acc_steps = 1

# optimizer
_C.optimizer = edict()
_C.optimizer.type = "SGD"
_C.optimizer.lr = 1e-2
_C.optimizer.momentum = 0.9
_C.optimizer.weight_decay = 1e-4

# scheduler
_C.scheduler = edict()
_C.scheduler.type = "Cosine"
_C.scheduler.total_steps = _C.trainer.max_epoch * _C.trainer.num_iters_per_epoch
_C.scheduler.warmup_steps = _C.trainer.num_iters_per_epoch
_C.scheduler.eta_init = 0.1
_C.scheduler.eta_min = 0.01

# model
_C.model = edict()
_C.model.num_stages = 5
_C.model.basic_voxel_size = 0.04
_C.model.kernel_size = 15
_C.model.kpconv_radius = 2.5
_C.model.kpconv_sigma = 2.0
_C.model.input_dim = 5
_C.model.init_dim = 64
_C.model.neighbor_limits = [24, 40, 34, 35, 34]  # For Area_5. Run 'calibrate_neighbors.py' for other areas.

# loss
_C.loss = edict()


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link_output", dest="link_output", action="store_true", help="link output dir")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = make_cfg()
    if args.link_output:
        os.symlink(cfg.exp.output_dir, "output")


if __name__ == "__main__":
    main()
