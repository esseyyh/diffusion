import torch 
import torch.nn as nn
from src.network.layers.embedding import SPE 


class DownBlock(nn.Module):
    def __init__(self,fan_in,fan_out,num_filters=3,time_embedding_dims=128):
        super().__init__()
        
        self.conv1 = nn.Conv2d(fan_in, fan_out, num_filters, padding=1)

        self.final = nn.Conv2d(fan_out, fan_out, 4, 2, 1)
        
        self.relu = nn.ReLU()
        
        self.bnorm1 = nn.BatchNorm2d(fan_out)
        
        self.bnorm2 = nn.BatchNorm2d(fan_out)
        
        self.conv2 = nn.Conv2d(fan_out, fan_out, 3, padding=1)
        
        self.time_emb = SPE(time_embedding_dims)
        
        self.mlp=nn.Linear(time_embedding_dims,fan_out)
    
    def forward(self,x,time): 
        
        x=self.bnorm1(self.relu(self.conv1(x)))
        
        x_time=self.relu( self.mlp (self.time_emb(time) ) )
        
        x= x + x_time[(..., ) + (None, ) * 2]
        
        x=self.bnorm2(self.relu(self.conv2(x)))
        
        return self.final(x)




