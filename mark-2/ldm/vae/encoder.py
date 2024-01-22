import torch
from torch import nn
from torch.nn import functional as F
from .builder import VAE_AttentionBlock, Block
from torchsummary import summary

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        

        self.layers=nn.Sequential(
            
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            Block(128, 128),
            
            Block(128, 128),
            
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            
            Block(128, 256), 
            
            
            Block(256, 256), 
            
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            
            
            Block(256, 512), 
            
            
            Block(512, 512), 
            
            
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            Block(512, 512), 
            
            
            Block(512, 512), 
            
            
            Block(512, 512), 
            
            
            VAE_AttentionBlock(512), 
            
            
            Block(512, 512), 
            
            nn.GroupNorm(32, 512), 
            
            
            nn.SiLU(), 

            
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 

            
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )
        

    def forward(self, x, noise=0.75):
        
        for module in self.layers:

            if getattr(module, 'stride', None) == (2, 2): 
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)
       
        log_variance = torch.clamp(log_variance, -30, 20)
        
        variance = log_variance.exp()
        
        stdev = variance.sqrt()
        
       
        x = mean + stdev * noise
        
       
        x *= 0.18215
        
        return x
    
