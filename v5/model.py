import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv2d, self).__init__()
        self.padding_time = (kernel_size[1] - 1) * dilation
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size, 
                              stride, 
                              padding=(0, self.padding_time), 
                              dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        if self.padding_time != 0:
            x = x[:, :, :, :-self.padding_time]
        return x
    

class CausalConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConvTranspose2d, self).__init__()
        self.padding_time = (kernel_size[1] - 1) * dilation
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding=0,
            dilation=dilation
        )
    def forward(self, x):
        x = self.conv_transpose(x)
        if self.padding_time != 0:
            x = x[:, :, :, :-self.padding_time]
        return x

class DownSample(nn.Module):
    def __init__(self, 
                 input_channel,
                 out_channel,
                 kernel_size,
                 stride,
                 activation,
                 normalization
                ):
        super().__init__()
        self.conv = nn.Sequential(
            CausalConv2d(input_channel, out_channel, 
                      kernel_size=kernel_size, stride=stride
                     ), 
            normalization, 
            activation
        )
    def forward(self, x):
        return self.conv(x)
    
class UpSample(nn.Module):
    def __init__(self, 
                 input_channel,
                 out_channel,
                 kernel_size,
                 stride,
                 activation,
                 normalization
                ):
        super().__init__()
        self.conv = nn.Sequential(
            CausalConvTranspose2d(input_channel, out_channel, 
                      kernel_size=kernel_size, stride=stride
                     ), 
            normalization, 
            activation
        )
    def forward(self, x1, x2):
        
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [0, 0, diffY // 2, diffY - diffY // 2] )
        x = torch.cat((x1,x2),1)
        
        return self.conv(x)
    
    
class Unet_model(nn.Module):
    def __init__(self, botlleneck=2):
        super().__init__()
        # enocder
        self.down_1 = DownSample(
                input_channel=1,
                out_channel=4, 
                kernel_size=(3, 3), 
                stride=(2, 1),
                activation = nn.GELU(),
                normalization = nn.BatchNorm2d(4)
            )
        self.down_2 = DownSample(
                input_channel=4,
                out_channel=8, 
                kernel_size=(3, 3), 
                stride=(2, 1),
                activation = nn.GELU(),
                normalization = nn.BatchNorm2d(8)
            )
        self.down_3 = DownSample(
                input_channel=8,
                out_channel=16, 
                kernel_size=(3, 3), 
                stride=(2, 1),
                activation = nn.GELU(),
                normalization = nn.BatchNorm2d(16)
            )
        self.down_4 = DownSample(
            input_channel=16, 
            out_channel=32, 
            kernel_size=(3,3), 
            stride=(2,1), 
            activation=nn.GELU(), 
            normalization=nn.BatchNorm2d(32)
        )
        
        self.down_5 = DownSample(
            input_channel=32, 
            out_channel=64, 
            kernel_size=(3,3), 
            stride=(2,1), 
            activation=nn.GELU(), 
            normalization=nn.BatchNorm2d(64)
        )
        self.down_6 = DownSample(
            input_channel=64, 
            out_channel=128, 
            kernel_size=(3,3), 
            stride=(2,1), 
            activation=nn.GELU(), 
            normalization=nn.BatchNorm2d(128)
        )
        # bottleneck
        self.lstm = nn.LSTM(input_size=128*botlleneck, 
                            hidden_size=128*botlleneck, 
                            num_layers=1, 
                            batch_first=True)
        # decoder
        self.up_0 = UpSample(
            input_channel=128*2, 
            out_channel=64, 
            kernel_size=(3,3), 
            stride=(2,1), 
            activation=nn.GELU(), 
            normalization=nn.BatchNorm2d(64)
        )
        
        self.up_1 = UpSample(
            input_channel=64*2, 
            out_channel=32, 
            kernel_size=(3,3), 
            stride=(2,1), 
            activation=nn.GELU(), 
            normalization=nn.BatchNorm2d(32)
        )
        self.up_2 = UpSample(
            input_channel=32*2, 
            out_channel=16, 
            kernel_size=(3,3), 
            stride=(2,1), 
            activation=nn.GELU(), 
            normalization=nn.BatchNorm2d(16)
        )
        self.up_3 = UpSample(
            input_channel=16*2, 
            out_channel=8, 
            kernel_size=(3,3), 
            stride=(2,1), 
            activation=nn.GELU(), 
            normalization=nn.BatchNorm2d(8)
        )
        self.up_4 = UpSample(
            input_channel=8*2, 
            out_channel=4, 
            kernel_size=(3,3), 
            stride=(2,1), 
            activation=nn.GELU(), 
            normalization=nn.BatchNorm2d(4)
        )
        self.up_5 = UpSample(
            input_channel=4*2, 
            out_channel=2, 
            kernel_size=(3,3), 
            stride=(2,1), 
            activation=nn.GELU(), 
            normalization=nn.BatchNorm2d(2)
        )
        self.convT = CausalConvTranspose2d(
            in_channels=2, 
            out_channels=1,
            kernel_size=(2,3),
            stride=(1,1),
        )
    def forward(self, x):
        B, C, F, T = x.shape # [B, 1, 200, T]
        
        x1 = self.down_1(x) # -> [B, 4, 99, T]
        x2 = self.down_2(x1) # -> [B, 8, 49, T]
        x3 = self.down_3(x2) # -> [B, 16, 24, T]
        x4 = self.down_4(x3) # -> [B, 32, 11, T]
        x5 = self.down_5(x4) # -> [B, 64, 5, T]
        x6 = self.down_6(x5) # -> [3, 128, 2, 400]
        
        _, C_c, F_c, _ = x6.shape
        x = x6.view(B, -1, T).permute(0, 2, 1)  # (B, C, F, T) -> (B, C*F, T) -> (B, T, C*F): [B, T, 640]
        x = self.lstm(x)[0]
        x = x.permute(0, 2, 1).view(B, C_c, F_c, T) # -> [B, 128, 5, T]

        x = self.up_0(x, x6) # -> [B, 64, 5, T]
        x = self.up_1(x, x5) # -> [B, 32, 11, T]
        x = self.up_2(x, x4) # -> [B, 16, 23, T]
        x = self.up_3(x, x3) # -> [B, 8, 49, T]
        x = self.up_4(x, x2) # -> [B, 4, 99, T]
        x = self.up_5(x, x1) # -> [B, 2, 199, T]
        return self.convT(x) # -> [B, 1, 200, T]