import torch
import torch.nn as nn
from src.network.layers.dwconv import DownBlock
from src.network.layers.upconv import UpBlock


class UNet(nn.Module):
    def __init__(self, img_channels = 3, time_embedding_dims = 128,  sequence_channels = (64, 128,256,512)):
        super().__init__()
        self.time_embedding_dims = time_embedding_dims
        #sequence_channels_rev = reversed(sequence_channels)
        
        self.downsampling = nn.ModuleList(
                [UpBlock(channels_in,
                channels_out,
                time_embedding_dims) for channels_in, channels_out in zip(sequence_channels, sequence_channels[1:])])
        
        self.upsampling = nn.ModuleList([DownBlock(channels_in,
                                               channels_out,
                                               time_embedding_dims) for channels_in, channels_out in zip(sequence_channels[::-1], sequence_channels[::-1][1:])])
        
        self.conv1 = nn.Conv2d(img_channels, sequence_channels[0], 3, padding=1)
        
        self.conv2 = nn.Conv2d(sequence_channels[0], img_channels, 1)

    
    def forward(self, x, t):

        # saving residuals for latter connection using concatination 
        residuals = []

        

        # initial convoltion before down sampling 
        out = self.conv1(x)
        
        for ds in self.downsampling:
            out = ds(out, t)
            #continued appending of the downsampled output 
            residuals.append(out)
        
        for us, res in zip(self.upsampling, reversed(residuals)):
            # concatinaing and upsampling 
            out = us(torch.cat((out, res), dim=1), t,)
            
        return self.conv2(out)
