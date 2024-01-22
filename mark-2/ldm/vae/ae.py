import torch 
import torch.nn as nn
from .decoder import Decoder
from .encoder import Encoder
from torchinfo import summary
from typing import List

class AE(nn.Module):
    


    def __init__(
        self,
        fan_in:int =3,
        bottle_neck: int =64,
        
        
    ):
        super().__init__()

    
        self.encoder=Encoder()
        self.decoder=Decoder()

    
    def decode(self, x):
        return self.decode(x)
    def encode(self, x):
        return self.encode(x)
    

    def forward(self, x):

        return self.decoder(self.encoder(x))
    
