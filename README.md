# A more easy-to-use implementation of KPConv

This repo contains a more easy-to-use implementation of KPConv based on PyTorch.

## Introduction

[KPConv](https://arxiv.org/abs/1904.08889) is a powerfull point convolution for point cloud processing. However, the original PyTorch implementation of [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch) has the following drawbacks:

1. It relies on heavy data preprocessing in the dataloader `collate_fn` to downsample the input point clouds, so one has to rewrite the `collate_fn` to work with KPConv. And the data processing is computed on CPU, which may be slow if the point clouds are large (e.g., KITTI).
2. The network architecture and the configurations of KPConv is fixed in the config file, and only single-branch FCN architecture is supported. For more complicated tasks, this is inflexible to build up multi-branch networks.

To use KPConv in more complicated networks, we build this repo with the following modifications:

1. GPU-based grid subsampling and radius neighbor searching. To accelerate kNN searching, we use [KeOps](https://github.com/getkeops/keops). This enables us to decouple grid subsampling with data loading.
2. Rebuilt KPConv interface. This enables us to insert KPConv anywhere in the network. All KPConv modules are rewritten to accept four inputs:
   1. `s_feats`: features of the support points.
   2. `q_points`: coordinates of the query points.
   3. `s_points`: coordinates of the support points.
   4. `neighbor_indices`: the indices of the neighbors for the query points.
3. [Group normalization](https://arxiv.org/abs/1803.08494) is used by default instead of [batch normalization](https://arxiv.org/abs/1502.03167). As point clouds are stacked in KPConv, BN is hard to implement. For this reason, we use GN instead.

More examples will be provided in the future.

## Acknowledgements

1. [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch)
2. [KeOps](https://github.com/getkeops/keops)
