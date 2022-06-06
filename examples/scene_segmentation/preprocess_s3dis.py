import os.path as osp
from glob import glob

import numpy as np
from tqdm import tqdm

from easy_kpconv.datasets.utils import grid_subsample
import easy_kpconv.datasets.s3dis as s3dis
from vision3d_engine.utils.io import dump_pickle, ensure_dir


raw_dataset_dir = "/path/to/Stanford3dDataset_v1.2"
dataset_dir = "/path/to/s3dis_semseg_raw_resolution"
save_dir = "/path/to/s3dis_semseg_voxelized_4cm"
ensure_dir(dataset_dir)
ensure_dir(save_dir)


def fix_x01_bug():
    print("Fixing x01 bug...")
    filenames = sorted(glob(osp.join(raw_dataset_dir, "*", "*", "Annotations", "*.txt")))
    for filename in tqdm(filenames):
        with open(filename, "r") as f:
            lines = f.readlines()

        require_fix = False
        for i, line in enumerate(lines):
            if "\x10" in line:
                require_fix = True
                lines[i] = line.replace("\x10", "0")

        if require_fix:
            print(f"{filename} fixed.")
            with open(filename, "w") as f:
                f.writelines(lines)
    print("Finished.")


def generate_data():
    print("Generating npz files...")
    scene_dirs = sorted(glob(osp.join(raw_dataset_dir, "*", "*")))
    scene_dirs = [scene_dir for scene_dir in scene_dirs if osp.isdir(scene_dir)]
    print(scene_dirs)
    scene_names = []
    pbar = tqdm(scene_dirs)
    for scene_dir in pbar:
        area_name = osp.basename(osp.dirname(scene_dir))
        room_name = osp.basename(scene_dir)
        scene_name = f"{area_name}-{room_name}"
        scene_names.append(scene_name)

        pbar.set_description(scene_name)

        points_list = []
        colors_list = []
        labels_list = []

        filenames = glob(osp.join(scene_dir, "Annotations", "*.txt"))
        for filename in filenames:
            class_name = osp.basename(filename).split("_")[0]
            if class_name not in s3dis.CLASS_NAMES:
                class_name = "clutter"
            class_id = s3dis.CLASS_NAMES.index(class_name)

            data = np.loadtxt(filename)
            points_list.append(data[:, :3])
            colors_list.append(data[:, 3:])
            labels_list.append(np.full(shape=(data.shape[0],), fill_value=class_id, dtype=np.int8))

        points = np.concatenate(points_list, axis=0)
        colors = np.concatenate(colors_list, axis=0) / 255.0
        labels = np.concatenate(labels_list, axis=0)

        output_filename = osp.join(dataset_dir, f"{scene_name}.npz")
        np.savez_compressed(output_filename, points=points, colors=colors, labels=labels)

    lines = [scene_name + "\n" for scene_name in scene_names]
    with open(osp.join(dataset_dir, "scene_names.txt"), "w") as f:
        f.writelines(lines)
    print("Finished.")


def voxelize():
    with open(osp.join(dataset_dir, "scene_names.txt")) as f:
        lines = f.readlines()
        scene_names = [line.strip() for line in lines]

    ensure_dir(save_dir)
    metadata_list = []
    for scene_name in tqdm(scene_names):
        filename = osp.join(dataset_dir, f"{scene_name}.npz")
        data_dict = np.load(filename)
        points = data_dict["points"]
        colors = data_dict["colors"]
        labels = data_dict["labels"].astype(np.int64)
        s_points, s_colors, s_labels, inv_indices = grid_subsample(points, colors, labels, 0.04, 13)
        filename = osp.join(save_dir, f"{scene_name}.npz")
        np.savez_compressed(
            filename, points=s_points, colors=s_colors, labels=s_labels, inv_indices=inv_indices, raw_labels=labels
        )
        metadata = {
            "area_name": scene_name.split("-")[0],
            "room_name": scene_name.split("-")[1],
            "scene_name": scene_name,
            "raw_points": points.shape[0],
            "vox_points": s_points.shape[0],
            "voxel_size": 0.04,
            "raw_file": f"raw_data/{scene_name}.npy",
            "vox_file": f"s3dis_semseg_voxelized_4cm/{scene_name}.npz",
        }
        metadata_list.append(metadata)
    dump_pickle(metadata_list, "/data/S3DIS/metadata.pkl")


if __name__ == "__main__":
    fix_x01_bug()
    generate_data()
    voxelize()
