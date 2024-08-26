import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Res1D(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride = 1, 
        padding = 0, 
        dilation = 1, 
        groups = 1,
        downsample = False,
        downsample_channels = False,
        bias = True, 
        padding_mode = "zeros", 
        device = None, 
        dtype = None
    ):

        super(Res1D, self).__init__()
        self.conv1d = MyConv1dPadSame(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            stride = stride, 
            groups = groups
        )

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.max_pool = MyMaxPool1dPadSame(kernel_size = stride)
        self.downsample = downsample
        self.downsample_channels = downsample_channels
        self.conv_res = MyConv1dPadSame(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            stride = stride, 
            groups = groups
        )

    def forward(self, x):
        out = self.relu(self.bn(self.conv1d(x)))

        if self.downsample and not self.downsample_channels:
            x = self.max_pool(x)

        if self.downsample and self.downsample_channels:
            x = self.conv_res(x)

        out += x
        return out

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net

class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net

class Net1D(nn.Module):

    def __init__(self, cfg = None):
        super(Net1D, self).__init__()
        self.cfg = cfg

        self.test = Res1D(
            in_channels = 1536, 
            out_channels = 768, 
            kernel_size = 8, 
            stride = 16, 
            groups = 1,
            downsample = True,
            downsample_channels=True
        )

        self.blockx = Res1D(
            in_channels = 1536, 
            out_channels = 1280, 
            kernel_size = 4, 
            stride = 1, 
            groups = 1,
            downsample = True,
            downsample_channels = True
        )

        self.blocky = Res1D(
            in_channels = 1280, 
            out_channels = 768, 
            kernel_size = 4, 
            stride = 1, 
            groups = 1,
            downsample = True,
            downsample_channels = True
        )

        self.block0 = Res1D(
            in_channels = 768, 
            out_channels = 768, 
            kernel_size = 4, 
            stride = 1, 
            groups = 1,
            downsample = False
        )

        self.block1 = Res1D(
            in_channels = 768, 
            out_channels = 768, 
            kernel_size = 4, 
            stride = 1, 
            groups = 1,
            downsample = False
        )

        self.block2 = Res1D(
            in_channels = 768, 
            out_channels = 768, 
            kernel_size = 2, 
            stride = 2, 
            groups = 1,
            downsample = True            
        )

        self.block3 = Res1D(
            in_channels = 768, 
            out_channels = 768, 
            kernel_size = 4, 
            stride = 1, 
            groups = 1,
            downsample = False            
        )

        self.block4 = Res1D(
            in_channels = 768, 
            out_channels = 768, 
            kernel_size = 2, 
            stride = 2, 
            groups = 1,
            downsample = True            
        )

        self.layer_f1 = torch.nn.Sequential(
            MyConv1dPadSame(
                in_channels = 768, 
                out_channels = 768, 
                kernel_size = 2, 
                stride = 2, 
                groups = 1
            ),
            nn.BatchNorm1d(768),
            nn.ReLU()
        )

        self.layer_f2 = torch.nn.Sequential(
            MyConv1dPadSame(
                in_channels = 768, 
                out_channels = 768, 
                kernel_size = 2, 
                stride = 2, 
                groups = 1
            ),
            nn.BatchNorm1d(768),
            nn.ReLU()
        )

        self.resx = MyConv1dPadSame(
            in_channels = 1536, 
            out_channels = 768, 
            kernel_size = 4,  
            groups = 1,
            stride = 2
        )

        self.resy = MyConv1dPadSame(
            in_channels = 768, 
            out_channels = 768, 
            kernel_size = 4,  
            groups = 1,
            stride = 4
        )

    def forward(self, x, y):

        x = torch.cat([x, y], dim = 1)
        x = self.test(x)
        return x

def main():

    input = torch.randn(1, 1536, 128)
    net1 = Net1D()
    out = net1(input)

    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    main()