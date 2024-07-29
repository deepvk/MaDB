import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv2d, self).__init__()
        self.padding_time = (kernel_size[1] - 1) * dilation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(0, self.padding_time), dilation=dilation)

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
    def forward(self, x):
        return self.conv(x)
    
class Unet_model(nn.Module):
    def __init__(
        self, input_channel=1, out_channel=32, complex_input=True, log_flag=True, subband_flag=False):
        super().__init__()
        self.complex_input = complex_input
        self.log_flag = log_flag
        self.subband_flag = subband_flag
        
        self.down_1 = DownSample(
                input_channel=8,
                out_channel=32, 
                kernel_size=(8, 3), 
                stride=(4, 1),
                activation = nn.GELU(),
                normalization = nn.BatchNorm2d(32)
            )
        
        self.down_2 = DownSample(
                input_channel=32,
                out_channel=64, 
                kernel_size=(8, 3), 
                stride=(4, 1),
                activation = nn.GELU(),
                normalization = nn.BatchNorm2d(64)
            )
        
        self.down_3 = DownSample(
                input_channel=64,
                out_channel=128, 
                kernel_size=(8, 3), 
                stride=(4, 1),
                activation = nn.GELU(),
                normalization = nn.BatchNorm2d(128)
            )
             
        self.lstm = nn.LSTM(input_size=128, 
                            hidden_size=128, 
                            num_layers=1, 
                            batch_first=True)
        
        
        self.up_1 = UpSample(
            input_channel=128+128,
            out_channel=64,
            kernel_size=(11, 1), 
            stride=(4, 1),
            activation = nn.GELU(), 
            normalization = nn.BatchNorm2d(64)
        )
        
        self.up_2 = UpSample(
            input_channel=64+64, 
            out_channel=32, 
            kernel_size=(9, 3), 
            stride=(4, 1),
            activation = nn.GELU(),
            normalization = nn.BatchNorm2d(32)
        )    
        
        self.up_3 = UpSample(
            input_channel=32+32,
            out_channel=16, 
            kernel_size=(8, 3), 
            stride=(4, 1),
            activation = nn.GELU(),
            normalization = nn.BatchNorm2d(16)
        )
        
        self.conv = nn.Sequential(
             nn.Conv2d(
                 in_channels=16, #16
                 out_channels=1, 
                 kernel_size=1, 
                 stride=1, 
                 padding=0
                     ),
            nn.BatchNorm2d(1),
            nn.GELU()
        )
        self.fc = nn.Sequential(
            nn.Linear(200, 256),
            nn.GELU(),
            nn.Linear(256, 200),
            nn.Sigmoid()
        )
    
    def cac2cws(self, x):
        k = 5
        b,c,f,t = x.shape
        x = x.reshape(b,c,k,f//k,t)
        x = x.reshape(b,c*k,f//k,t)
        return x
    
    def cws2cac(self, x):
        k = 5
        b,c,f,t = x.shape
        x = x.reshape(b,c//k,k,f,t)
        x = x.reshape(b,c//k,f*k,t)
        return x
    
    def forward(self, x):
        B, C, F, T = x.shape
        
        if self.complex_input:
            x = torch.concat([x.real, x.imag], dim=1)
        elif self.log_flag:
            x = torch.log(x + 1e-5)
        
#         if self.subband_flag:
#             x = cac2cws(x)
        #print(x.shape)
        x1 = self.down_1(x)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        
        _, C_c, F_c, _ = x3.shape
        x = x3.view(B, -1, T).permute(0, 2, 1) # (B, C, F, T) -> (B, C*F, T) -> (B, T, C*F)
        x = self.lstm(x)[0] + x
        x = x.permute(0, 2, 1).view(B, C_c, F_c, T)
        
        x = self.up_1(torch.concat([x,x3],dim=1))
        x = self.up_2(torch.concat([x,x2],dim=1))
        x = self.up_3(torch.concat([x,x1],dim=1))
        x = self.conv(x)
        
        x = x.view(B, -1, T).permute(0, 2, 1)
        x = self.fc(x)
        
        return x.view(B, 1, F, T)