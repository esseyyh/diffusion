import torch
from torch import nn
from torch.nn import functional as F
from .builder import SelfAttention




class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embd)                                             # A matrix that lerns the embedding position of words
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):
       
        x = self.token_embedding(tokens)                                                                 # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim) 
        x += self.position_embedding                                                                     # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        
        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
       
        self.layernorm_1 = nn.LayerNorm(n_embd)
      
        self.attention = SelfAttention(n_head, n_embd)

        self.layernorm_2 = nn.LayerNorm(n_embd)
    
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        
        residue = x
        
        ### SELF ATTENTION ###

        
        x = self.layernorm_1(x)                                                               # layer norm before multi headed attention
       
        x = self.attention(x, causal_mask=True)                                               # multi headed attention 
        
        x += residue                                                                          # residual skip connection

        ### FEEDFORWARD LAYER ###
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension. 

        residue = x
        
        x = self.layernorm_2(x)                                                                # layer norm followed by linear layer -> activation -> linear layer  
        
        x = self.linear_1(x)
        
        x = x * torch.sigmoid(1.702 * x)                                                        # GELU activation function
        
       
        x = self.linear_2(x)
        
        
        x += residue                                                                            # final  skip connection

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        
        state = self.embedding(tokens)                                                          # embeded the tokens 

        
        for layer in self.layers:                                                               # 12 sequential transformer encoder layers (multihead attention layers )
            
            state = layer(state)
        
        output = self.layernorm(state)                                                          # final activation 
        
        return output