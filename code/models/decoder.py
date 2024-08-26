import torch

class Decoder(torch.nn.Module):
    def __init__(self, cfg = None):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 256, kernel_size = (4, 2, 2), stride = (4, 2, 2), bias=True, padding=0),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size = (4, 2, 2), stride = (4, 2, 2), bias=True, padding=0),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size = (4, 2, 2), stride = (4, 2, 2), bias=True, padding=0),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 64, kernel_size = (1, 2, 2), stride = (1, 2, 2), bias=True, padding=0),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 3, kernel_size = (2, 2, 2), stride = (2, 2, 2), bias=True, padding=0),
            torch.nn.Tanh()
        )

    def forward(self, feature_map):

        feature_map = feature_map.unsqueeze(2)
        feature = self.layer1(feature_map)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)
        images = self.layer5(feature)
        return images