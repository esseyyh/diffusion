import torch
import torch.nn as nn 
import torch.nn.functional as F
import math





class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        """a self attention layer with an opional casual mask  """     

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):



        
        input_shape = x.shape                                                         # input image (x ) size  (Batch, H*W ,Channels )
        batch_size, sequence_length, d_embed = input_shape 
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)      # dividing the channel dim to allow for multiple heads in our case the number of heads is likly one 
        q, k, v = self.in_proj(x).chunk(3, dim=-1)                                    # pass the vector through a linear projeciton and use chunk  to get  key query and vector outs 
        q = q.view(interim_shape).transpose(1, 2)                                     # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

       
        weight = q @ k.transpose(-1, -2)                                                # (Batch_Size, H, Seq_Len, Dim) @ (Batch_Size, H, Dim, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        
        if causal_mask:                                                                  # if casual mask is applied the weights matrix (batch , n_head , seq, seq ) a mask is applied above diagonal 
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)                     # create a tensor of ones same shape as weights 
            weight.masked_fill_(mask, -torch.inf)                                        # Fill the upper triangle with -inf so that after softmax it becomes zeros  masking the next input 
        
        
        weight /= math.sqrt(self.d_head)                                                # divide by  (Dim / H) for scaling 
        weight = F.softmax(weight, dim=-1)                                              # soft max to get the probability distribution 
        output = weight @ v                                                             # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim ) -> (Batch_Size, H, Seq_Len, Dim)
        output = output.transpose(1, 2)                                                 # (Batch_Size, H, Seq_Len, Dim ) -> (Batch_Size, Seq_Len, H, Dim)

        
        output = output.reshape(input_shape)                                            # (Batch_Size, Seq_Len, H, Dim ) -> (Batch_Size, Seq_Len, Dim)
                                              
        
       
        return self.out_proj(output)

