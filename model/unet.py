import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels =  1, out_channels = 1, padding = 1) -> None:
        super().__init__()

        self.sequantial = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size= 3, padding= padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels= out_channels, kernel_size= 3, padding= padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.sequantial(x)
    

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, x):
        x = self.conv(x)
        return x, self.pool(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.upConv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size= 2, stride= 2, padding= 0)
        self.conv = ConvBlock(2*out_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upConv(x)
        x = torch.cat([x, skip], axis = 1)
        x = self.conv(x) 
        return x



class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        '''
        Initializes the U-Net model, defining the encoder, decoder, and other layers.

        Args:
        - in_channels (int): Number of input channels (1 for scan images).
        - out_channels (int): Number of output channels (1 for binary segmentation masks).
        
        Function:
        - CBR (in_channels, out_channels): Helper function to create a block of Convolution-BatchNorm-ReLU layers. 
        (This function is optional to use)
        '''
        super(UNet, self).__init__()
        
        self.enc1 = DownSample(in_channels,32)
        self.enc2 = DownSample(32,64)
        self.enc3 = DownSample(64,128)
        self.enc4 = DownSample(128,256)

        self.bott = ConvBlock(256,512)

        self.decod1 = UpSample(512,256)
        self.decod2 = UpSample(256,128)
        self.decod3 = UpSample(128,64)
        self.decod4 = UpSample(64,32)

        self.output = nn.Conv2d(32,out_channels,kernel_size=1, padding= 0)
        
    
    def forward(self, x):
        '''
        Defines the forward pass of the U-Net, performing encoding, bottleneck, and decoding operations.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        '''
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b = self.bott(p4)

        d1 = self.decod1(b, s4)
        d2 = self.decod2(d1, s3)
        d3 = self.decod3(d2, s2)
        d4 = self.decod4(d3, s1)

        output = self.output(d4)
        return output