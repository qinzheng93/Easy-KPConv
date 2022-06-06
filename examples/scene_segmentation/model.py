import torch
import torch.nn as nn

from easy_kpconv.layers.kpconv_blocks import KPConvBlock, KPResidualBlock
from easy_kpconv.layers.unary_block import UnaryBlockPackMode
from easy_kpconv.ops.graph_pyramid import build_grid_and_radius_graph_pyramid
from easy_kpconv.ops.nearest_interpolate import nearest_interpolate_pack_mode


class KPFCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_stages = cfg.model.num_stages
        self.voxel_size = cfg.model.basic_voxel_size
        self.kpconv_radius = cfg.model.kpconv_radius
        self.kpconv_sigma = cfg.model.kpconv_sigma
        self.neighbor_limits = cfg.model.neighbor_limits
        self.first_radius = self.voxel_size * self.kpconv_radius
        self.first_sigma = self.voxel_size * self.kpconv_sigma

        input_dim = cfg.model.input_dim
        first_dim = cfg.model.init_dim
        kernel_size = cfg.model.kernel_size
        first_radius = self.first_radius
        first_sigma = self.first_sigma

        self.encoder1_1 = KPConvBlock(input_dim, first_dim, kernel_size, first_radius, first_sigma)
        self.encoder1_2 = KPResidualBlock(first_dim, first_dim * 2, kernel_size, first_radius, first_sigma)

        self.encoder2_1 = KPResidualBlock(
            first_dim * 2, first_dim * 2, kernel_size, first_radius, first_sigma, strided=True
        )
        self.encoder2_2 = KPResidualBlock(first_dim * 2, first_dim * 4, kernel_size, first_radius * 2, first_sigma * 2)
        self.encoder2_3 = KPResidualBlock(first_dim * 4, first_dim * 4, kernel_size, first_radius * 2, first_sigma * 2)

        self.encoder3_1 = KPResidualBlock(
            first_dim * 4, first_dim * 4, kernel_size, first_radius * 2, first_sigma * 2, strided=True
        )
        self.encoder3_2 = KPResidualBlock(first_dim * 4, first_dim * 8, kernel_size, first_radius * 4, first_sigma * 4)
        self.encoder3_3 = KPResidualBlock(first_dim * 8, first_dim * 8, kernel_size, first_radius * 4, first_sigma * 4)

        self.encoder4_1 = KPResidualBlock(
            first_dim * 8, first_dim * 8, kernel_size, first_radius * 4, first_sigma * 4, strided=True
        )
        self.encoder4_2 = KPResidualBlock(first_dim * 8, first_dim * 16, kernel_size, first_radius * 8, first_sigma * 8)
        self.encoder4_3 = KPResidualBlock(
            first_dim * 16, first_dim * 16, kernel_size, first_radius * 8, first_sigma * 8
        )

        self.encoder5_1 = KPResidualBlock(
            first_dim * 16, first_dim * 16, kernel_size, first_radius * 8, first_sigma * 8, strided=True
        )
        self.encoder5_2 = KPResidualBlock(
            first_dim * 16, first_dim * 32, kernel_size, first_radius * 16, first_sigma * 16
        )
        self.encoder5_3 = KPResidualBlock(
            first_dim * 32, first_dim * 32, kernel_size, first_radius * 16, first_sigma * 16
        )

        self.decoder4 = UnaryBlockPackMode(first_dim * 48, first_dim * 16)
        self.decoder3 = UnaryBlockPackMode(first_dim * 24, first_dim * 8)
        self.decoder2 = UnaryBlockPackMode(first_dim * 12, first_dim * 4)
        self.decoder1 = UnaryBlockPackMode(first_dim * 6, first_dim * 2)

        self.classifier = nn.Sequential(
            nn.Linear(first_dim * 2, first_dim),
            nn.GroupNorm(8, first_dim),
            nn.ReLU(),
            nn.Linear(first_dim, cfg.data.num_classes),
        )

    def forward(self, data_dict):
        output_dict = {}

        feats = data_dict["feats"]
        points = data_dict["points"]
        lengths = data_dict["lengths"]

        graph_pyramid = build_grid_and_radius_graph_pyramid(
            points, lengths, self.num_stages, self.voxel_size, self.first_radius, self.neighbor_limits
        )

        points_list = graph_pyramid["points"]
        neighbors_list = graph_pyramid["neighbors"]
        subsampling_list = graph_pyramid["subsampling"]
        upsampling_list = graph_pyramid["upsampling"]

        feats_s1 = torch.cat([torch.ones_like(feats[:, :1]), feats], dim=1)
        feats_s1 = self.encoder1_1(points_list[0], points_list[0], feats_s1, neighbors_list[0])
        feats_s1 = self.encoder1_2(points_list[0], points_list[0], feats_s1, neighbors_list[0])

        feats_s2 = self.encoder2_1(points_list[1], points_list[0], feats_s1, subsampling_list[0])
        feats_s2 = self.encoder2_2(points_list[1], points_list[1], feats_s2, neighbors_list[1])
        feats_s2 = self.encoder2_3(points_list[1], points_list[1], feats_s2, neighbors_list[1])

        feats_s3 = self.encoder3_1(points_list[2], points_list[1], feats_s2, subsampling_list[1])
        feats_s3 = self.encoder3_2(points_list[2], points_list[2], feats_s3, neighbors_list[2])
        feats_s3 = self.encoder3_3(points_list[2], points_list[2], feats_s3, neighbors_list[2])

        feats_s4 = self.encoder4_1(points_list[3], points_list[2], feats_s3, subsampling_list[2])
        feats_s4 = self.encoder4_2(points_list[3], points_list[3], feats_s4, neighbors_list[3])
        feats_s4 = self.encoder4_3(points_list[3], points_list[3], feats_s4, neighbors_list[3])

        feats_s5 = self.encoder5_1(points_list[4], points_list[3], feats_s4, subsampling_list[3])
        feats_s5 = self.encoder5_2(points_list[4], points_list[4], feats_s5, neighbors_list[4])
        feats_s5 = self.encoder5_3(points_list[4], points_list[4], feats_s5, neighbors_list[4])

        latent_s5 = feats_s5

        latent_s4 = nearest_interpolate_pack_mode(latent_s5, upsampling_list[3])
        latent_s4 = torch.cat([latent_s4, feats_s4], dim=1)
        latent_s4 = self.decoder4(latent_s4)

        latent_s3 = nearest_interpolate_pack_mode(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)

        latent_s2 = nearest_interpolate_pack_mode(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)

        latent_s1 = nearest_interpolate_pack_mode(latent_s2, upsampling_list[0])
        latent_s1 = torch.cat([latent_s1, feats_s1], dim=1)
        latent_s1 = self.decoder1(latent_s1)

        scores = self.classifier(latent_s1)

        output_dict["scores"] = scores

        return output_dict


def create_model(cfg):
    return KPFCNN(cfg)


def run_test():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == "__main__":
    run_test()
