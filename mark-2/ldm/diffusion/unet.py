import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary 

from .builder import res_block , Attentionblock,Sequential,Upsample


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
    



x=torch.randn(1,4,64,64)

time=torch.randn(1,1280)

context=torch.randn(1, 77, 768)

blok=UNET()

summary(blok,input_data=[x,context,time])

print(blok(x,context,time).shape)

