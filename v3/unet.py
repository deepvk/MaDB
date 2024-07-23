import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class DownSample(nn.Module):
    def __init__(self, 
                 input_channel,
                 out_channel,
                 kernel_size,
                 stride,
                 padding,
                 activation,
                 normalization
                ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, out_channel, 
                      kernel_size=kernel_size, stride=stride, padding=padding
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
            nn.ConvTranspose2d(input_channel, out_channel, 
                      kernel_size=kernel_size, stride=stride
                     ), 
            normalization, 
            activation
        )
    def forward(self, x):
        return self.conv(x)


class Unet_model(nn.Module):
    def __init__(
        self, input_channel=1, out_channel=32, log_flag=True):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.log_flag = log_flag
        
        self.down_1 = DownSample(
                input_channel=1,
                out_channel=32, 
                kernel_size=(8, 1), 
                stride=(4, 1),
                padding = (0, 0),
                activation = nn.GELU(),
                normalization = nn.BatchNorm2d(32)
            )
        
        self.down_2 = DownSample(
                input_channel=32,
                out_channel=64, 
                kernel_size=(8, 1), 
                stride=(4, 1),
                padding = (0, 0),
                activation = nn.GELU(),
                normalization = nn.BatchNorm2d(64)
            )
        
        self.down_3 = DownSample(
                input_channel=64,
                out_channel=128, 
                kernel_size=(8, 1), 
                stride=(4, 1),
                padding = (0, 0),
                activation = nn.GELU(),
                normalization = nn.BatchNorm2d(128)
            )
             
        self.lstm = nn.LSTM(input_size=128, 
                            hidden_size=128, 
                            num_layers=1, 
                            batch_first=True)
        
        
        self.up_1 = UpSample(
            input_channel=128,
            out_channel=64,
            kernel_size=(11, 1), 
            stride=(4, 1),
            activation = nn.GELU(), 
            normalization = nn.BatchNorm2d(64)
        )
        
        self.up_2 = UpSample(
            input_channel=64, 
            out_channel=32, 
            kernel_size=(9, 1), 
            stride=(4, 1),
            activation = nn.GELU(),
            normalization = nn.BatchNorm2d(32)
        )    
        
        self.up_3 = UpSample(
            input_channel=32,
            out_channel=16, 
            kernel_size=(8, 1), 
            stride=(4, 1),
            activation = nn.GELU(),
            normalization = nn.BatchNorm2d(16)
        )
        
        self.conv = nn.Sequential(
             nn.Conv2d(
                 in_channels=16, #16
                 out_channels=2, 
                 kernel_size=1, 
                 stride=1, 
                 padding=0
                     ),
            nn.InstanceNorm2d(2),
            nn.GELU()
        )
        self.fc = nn.Sequential(
            nn.Linear(400, 256),
            nn.InstanceNorm2d(400),
            nn.GELU(),
            nn.Linear(256, 200),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, F, T = x.shape
        
        x = x.abs().mean(dim=1).view(B, 1, F, T)  # (B, F, T) -> (B, 1, F, T)
        if self.log_flag:
            x = torch.log(x + 1e-5)


        x1 = self.down_1(x)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        
        _, C_c, F_c, _ = x3.shape
        x = x3.view(B, -1, T).permute(0, 2, 1) # (B, C, F, T) -> (B, C*F, T) -> (B, T, C*F)
        x = self.lstm(x)[0] + x
        x = x.permute(0, 2, 1).view(B, C_c, F_c, T)
        
        x = self.up_1(x + x3)
        x = self.up_2(x + x2)
        x = self.up_3(x + x1)
        
        x = self.conv(x)
        x = x.view(B, -1, T).permute(0, 2, 1)
        x = self.fc(x)
        return x.view(B, 1, F, T)
