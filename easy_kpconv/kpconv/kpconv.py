import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel_points import load_kernels
from ...utils.torch_utils import index_select


class KPConv(nn.Module):
    def __init__(
            self,
            kernel_size,
            input_dim,
            output_dim,
            radius,
            sigma,
            dimension=3,
            bias=False,
            inf=1e6,
            eps=1e-9
    ):
        """Initialize parameters for KPConv.

        Modified from [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

        Deformable KPConv is not supported.

        Args:
             kernel_size: Number of kernel points.
             input_dim: dimension of input features.
             output_dim: dimension of output features.
             radius: radius used for kernel point init.
             sigma: influence radius of each kernel point.
             dimension: dimension of the point space.
             bias: use bias or not (default: False)
             inf: value of infinity to generate the padding point
             eps: epsilon for gaussian influence
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.radius = radius
        self.sigma = sigma
        self.dimension = dimension
        self.has_bias = bias

        self.inf = inf
        self.eps = eps

        # Initialize weights
        self.weights = nn.Parameter(torch.zeros(self.kernel_size, input_dim, output_dim))
        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(self.output_dim))

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.register_buffer('kernel_points', self.initialize_kernel_points())

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.has_bias:
            nn.init.zeros_(self.bias)

    def initialize_kernel_points(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        kernel_points = load_kernels(
            self.radius, self.kernel_size, dimension=self.dimension, fixed='center'
        )
        return torch.from_numpy(kernel_points).float()

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        # Add a fake point in the last row for shadow neighbors
        s_points = torch.cat([s_points, torch.zeros_like(s_points[:1, :]) + self.inf], 0)
        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_points[neighbor_indices, :]
        # Center every neighborhood
        neighbors = neighbors - q_points.unsqueeze(1)
        # Get all difference matrices [n_points, n_neighbors, n_kernel_points, dim]
        neighbors = neighbors.unsqueeze(2)
        differences = neighbors - self.kernel_points
        # Get the square distances [n_points, n_neighbors, n_kernel_points]
        sq_distances = torch.sum(differences ** 2, dim=3)

        # Get Kernel point influences [n_points, n_kernel_points, n_neighbors]
        # Influence decrease linearly with the distance, and get to zero when d = sigma.
        neighbor_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.sigma, min=0.0)
        neighbor_weights = torch.transpose(neighbor_weights, 1, 2)

        # Add a zero feature for shadow neighbors
        s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, input_dim]
        neighbor_feats = index_select(s_feats, neighbor_indices, dim=0)

        # Apply distance weights [n_points, n_kernel_points, input_dim]
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feats)

        # Apply network weights [n_kernel_points, n_points, output_dim]
        weighted_feats = weighted_feats.permute((1, 0, 2))
        kernel_outputs = torch.matmul(weighted_feats, self.weights)

        # Convolution sum [n_points, output_dim]
        output_feats = torch.sum(kernel_outputs, dim=0, keepdim=False)

        # normalization term.
        neighbor_feats_sum = torch.sum(neighbor_feats, dim=-1)
        neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.), dim=-1)
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
        output_feats = output_feats / neighbor_num.unsqueeze(1)

        # bias term
        if self.has_bias:
            output_feats = output_feats + self.bias

        return output_feats

    def __repr__(self):
        repr_str = self.__class__.__name__ + '('
        repr_str += 'kernel_size: {}, input_dim: {}, output_dim: {}, radius: {:.2f}, sigma: {:.2f}, bias: {}'.format(
            self.kernel_size, self.input_dim, self.output_dim, self.radius, self.sigma, self.has_bias
        )
        repr_str += ')'
        return repr_str
