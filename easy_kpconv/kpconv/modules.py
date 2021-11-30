import torch
import torch.nn as nn

from .functional import maxpool
from .kpconv import KPConv


class GroupNorm(nn.Module):
    def __init__(self, num_group, feature_dim):
        r"""Initialize a group normalization block.

        Args:
            input_dim: feature dimension
            num_group: number of groups
        """
        super(GroupNorm, self).__init__()
        self.num_group = num_group
        self.feature_dim = feature_dim
        self.norm = nn.GroupNorm(self.num_group, self.feature_dim)

    def forward(self, x):
        x = x.transpose(0, 1).unsqueeze(0)  # (N, C) -> (B, C, N)
        x = self.norm(x)
        x = x.squeeze(0).transpose(0, 1)  # (B, C, N) -> (N, C)
        return x.squeeze()

    def __repr__(self):
        return self.__class__.__name__ + '(num_group: {}, feature_dim: {})'.format(self.num_group, self.feature_dim)


class UnaryBlock(nn.Module):
    def __init__(self, input_dim, output_dim, group_norm, has_relu=True):
        r"""Initialize a standard unary block with GroupNorm and LeakyReLU.

        Args:
            input_dim: dimension input features
            output_dim: dimension input features
            group_norm: number of groups in group normalization (None if group norm is not used)
        """
        super(UnaryBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.group_norm = group_norm
        self.has_relu = has_relu
        self.mlp = nn.Linear(input_dim, output_dim, bias=False)
        self.norm = GroupNorm(group_norm, output_dim)
        if self.has_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        if self.has_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(input_dim: {}, output_dim: {}, group_norm: {}, has_relu: {})'.format(
            self.input_dim, self.output_dim, self.group_norm, self.has_relu
        )


class LastUnaryBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        r"""Initialize a standard last_unary block without GN, ReLU.

        Args:
            input_dim: dimension input features
            output_dim: dimension input features
        """
        super(LastUnaryBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(input_dim: {}, output_dim: {})'.format(self.input_dim, self.output_dim)


class ConvBlock(nn.Module):
    def __init__(
            self,
            kernel_size,
            input_dim,
            output_dim,
            radius,
            sigma,
            group_norm
    ):
        r"""Initialize a KPConv block with ReLU and BatchNorm.

        Args:
            kernel_size: number of kernel points
            input_dim: dimension input features
            output_dim: dimension input features
            radius: convolution radius
            sigma: influence radius of each kernel point
            group_norm: group number for GroupNorm
        """
        super(ConvBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.KPConv = KPConv(kernel_size, input_dim, output_dim, radius, sigma, bias=False)
        self.norm = GroupNorm(group_norm, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        x = self.KPConv(s_feats, q_points, s_points, neighbor_indices)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
            self,
            kernel_size,
            input_dim,
            output_dim,
            radius,
            sigma,
            group_norm,
            strided=False
    ):
        r"""Initialize a ResNet bottleneck block.

        Args:
            kernel_size: number of kernel points
            input_dim: dimension input features
            output_dim: dimension input features
            radius: convolution radius
            sigma: influence radius of each kernel point
            group_norm: group number for GroupNorm
            strided: strided or not
        """
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.strided = strided

        hidden_dim = output_dim // 4

        if input_dim != hidden_dim:
            self.unary1 = UnaryBlock(input_dim, hidden_dim, group_norm)
        else:
            self.unary1 = nn.Identity()

        self.KPConv = KPConv(kernel_size, hidden_dim, hidden_dim, radius, sigma, bias=False)
        self.norm_conv = GroupNorm(group_norm, hidden_dim)

        self.unary2 = UnaryBlock(hidden_dim, output_dim, group_norm, has_relu=False)

        if input_dim != output_dim:
            self.unary_shortcut = UnaryBlock(input_dim, output_dim, group_norm, has_relu=False)
        else:
            self.unary_shortcut = nn.Identity()

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        x = self.unary1(s_feats)

        x = self.KPConv(x, q_points, s_points, neighbor_indices)
        x = self.norm_conv(x)
        x = self.leaky_relu(x)

        x = self.unary2(x)

        if self.strided:
            shortcut = maxpool(s_feats, neighbor_indices)
        else:
            shortcut = s_feats
        shortcut = self.unary_shortcut(shortcut)

        x = x + shortcut
        x = self.leaky_relu(x)

        return x
