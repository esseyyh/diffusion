import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary 

from .builder import res_block , Attentionblock,Sequential,Upsample,TimeEmbedding



class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            Sequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            Sequential(res_block(320,320),Attentionblock(8,40)),
            
            Sequential(res_block(320,320),Attentionblock(8,40)),
            
            Sequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            Sequential(res_block(320, 640), Attentionblock(8, 80)),
            
            Sequential(res_block(640, 640), Attentionblock(8, 80)),
            
            Sequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            Sequential(res_block(640, 1280), Attentionblock(8, 160)),
            
            Sequential(res_block(1280, 1280), Attentionblock(8, 160)),
            
            Sequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            Sequential(res_block(1280, 1280),res_block(1280, 1280)),
            
            Sequential(res_block(1280, 1280),res_block(1280, 1280))

        ])

        self.mid=nn.ModuleList([
            res_block(1280, 1280),

            res_block(1280, 1280),
            
            res_block(1280, 1280),


        ])


        self.decoders = nn.ModuleList([

            Sequential(res_block(2560,1280)),

            Sequential(res_block(2560,1280)),
            
            Sequential(res_block(2560,1280),Upsample(1280)),

            Sequential(res_block(2560,1280),Attentionblock(8,160)),

            Sequential(res_block(2560,1280),Attentionblock(8,160)),
            
            Sequential(res_block(1920,1280),Attentionblock(8,160),Upsample(1280)),
            
            Sequential(res_block(1920,640),Attentionblock(8,80)),

            Sequential(res_block(1280,640),Attentionblock(8,80)),

            Sequential(res_block(960,640),Attentionblock(8,80),Upsample(640)),

            Sequential(res_block(960,320),Attentionblock(8,40)),

            Sequential(res_block(640,320),Attentionblock(8,40)),

            Sequential(res_block(640,320),Attentionblock(8,40)),

        ])






           

    def forward(self, x, context, time):
 


        """accepts the latent image, prompt and time step to denoise the image a single step implemetation """
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        res = []


        for layer in self.encoders:
            x=layer(x,context,time)
            res.append(x)

        for layer in self.mid:
            x=layer(x,time)

        for layers in self.decoders:
           
            x = torch.cat((x, res.pop()), dim=1) 
            x = layers(x, context, time)

        


        
        return x
    

    
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        
        x = self.groupnorm(x)                                                                                # a simple conv layer for changing the dimension of the latent from diffusion , x: group norm -> activation -> conv 

        
        
        x = F.silu(x)
        
       
        x = self.conv(x)
        
        
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent, context, time):
        
        time = self.time_embedding(time)                                                                        # diffusion a wrapper around the unet to embeed the time using sin and cos and suppling the output to the unet 
        
        
        output = self.unet(latent, context, time)
        
        
        output = self.final(output)
        
        
        return output
    

