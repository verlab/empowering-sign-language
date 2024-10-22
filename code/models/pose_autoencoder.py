import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def ConvNormRelu(in_channels, out_channels, downsample=False, padding=0, batchnorm=True):
    if not downsample:
        k = 3
        s = 1
    else:
        k = 4
        s = 2

    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
    norm_block = nn.BatchNorm1d(out_channels)

    if batchnorm:
        net = nn.Sequential(
            conv_block,
            norm_block,
            nn.LeakyReLU(0.2, True)
        )
    else:
        net = nn.Sequential(
            conv_block,
            nn.LeakyReLU(0.2, True)
        )

    return net


class PoseEncoderConv(nn.Module):
    def __init__(self, length, dim):
        super().__init__()

        self.net = nn.Sequential(
            ConvNormRelu(dim, 32, batchnorm=True),
            ConvNormRelu(32, 64, batchnorm=True),
            ConvNormRelu(64, 64, True, batchnorm=True),
            nn.Conv1d(64, 32, 3)
        )


        if length == 64:
            self.out_net = nn.Sequential(
               nn.Linear(864, 256),  # for 64 frames
               #nn.Linear(384, 256),  # for 34 frames
               nn.BatchNorm1d(256),
               nn.LeakyReLU(True),
               nn.Linear(256, 128),
               nn.BatchNorm1d(128),
               nn.LeakyReLU(True),
               nn.Linear(128, 32),
            )
        elif length == 34:
            self.out_net = nn.Sequential(
               #nn.Linear(864, 256),  # for 64 frames
               nn.Linear(384, 256),  # for 34 frames
               nn.BatchNorm1d(256),
               nn.LeakyReLU(True),
               nn.Linear(256, 128),
               nn.BatchNorm1d(128),
               nn.LeakyReLU(True),
               nn.Linear(128, 32),
            )
        else:
            assert False

        self.fc_mu = nn.Linear(32, 32)
        self.fc_logvar = nn.Linear(32, 32)

    def forward(self, poses, variational_encoding):
        # encode
        poses = poses.transpose(1, 2)  # to (bs, dim, seq)
        out = self.net(poses)
        out = out.flatten(1)
        out = self.out_net(out)

        # return out, None, None
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        if variational_encoding:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar


class PoseDecoderConv(nn.Module):
    def __init__(self, length, dim, use_pre_poses=False):
        super().__init__()
        self.use_pre_poses = use_pre_poses

        feat_size = 32
        if use_pre_poses:
            self.pre_pose_net = nn.Sequential(
                nn.Linear(dim * 4, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
            feat_size += 32

        if length == 64:
            self.pre_net = nn.Sequential(
                nn.Linear(feat_size, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(True),
                nn.Linear(128, 256),
            )
        elif length == 34:
            self.pre_net = nn.Sequential(
                nn.Linear(feat_size, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(True),
                nn.Linear(64, 136),
            )
        else:
            assert False

        self.net = nn.Sequential(
            nn.ConvTranspose1d(4, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 32, 3),
            nn.Conv1d(32, dim, 3),
        )

    def forward(self, feat, pre_poses=None):
        if self.use_pre_poses:
            pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
            feat = torch.cat((pre_pose_feat, feat), dim=1)

        out = self.pre_net(feat)
        out = out.view(feat.shape[0], 4, -1)
        out = self.net(out)
        out = out.transpose(1, 2)
        return out



class EmbeddingNet(nn.Module):
    def __init__(self, pose_dim, n_frames):
        super().__init__()
        self.pose_encoder = PoseEncoderConv(n_frames, pose_dim)
        self.decoder = PoseDecoderConv(n_frames, pose_dim)
       
    def forward(self, poses, variational_encoding=False):
        
        poses_feat, pose_mu, pose_logvar = self.pose_encoder(poses, variational_encoding)
            # poses_feat = F.normalize(poses_feat, p=2, dim=1)
        
        # decoder
       
        latent_feat = poses_feat
        
        out_poses = self.decoder(latent_feat)

        return poses_feat, pose_mu, pose_logvar, out_poses

    def freeze_pose_nets(self):
        for param in self.pose_encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    # for model debugging
    n_frames = 64
    pose_dim = 138
    encoder = PoseEncoderConv(n_frames, pose_dim)
    decoder = PoseDecoderConv(n_frames, pose_dim)

    poses = torch.randn((128, n_frames, pose_dim))
    feat, _, _ = encoder(poses, True)
    recon_poses = decoder(feat)

    print('input', poses.shape)
    print('feat', feat.shape)
    print('output', recon_poses.shape)

    generator = EmbeddingNet(pose_dim, n_frames)
    feat, a, b, recon_poses = generator(poses)
   
    print('input', poses.shape)
    print('feat', feat.shape)
    print('output', recon_poses.shape)


