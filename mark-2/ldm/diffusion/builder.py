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

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        """a multi head attention layer that pays attnetion of the prompt to image """
        super().__init__()
        
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):

        ## the input to this layer is an a image latent and a prompt tokens 
        
                                                                                                # context:  (Batch_, Seq_Len, Dim) = (Batch_Size, 77, 768)

        input_shape = x.shape                                                                   # latent :  (Batch, H * W,  Channels)
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        
        q = self.q_proj(x)                                                                      # create the query tensor of the image
        k = self.k_proj(y)                                                                      # key vector of the prompt   
        v = self.v_proj(y)                                                                      # value vector of the prompt 

        
        q = q.view(interim_shape).transpose(1, 2)                                               # generate the query from the image while the key and values are geenrated from the prompt
        
        k = k.view(interim_shape).transpose(1, 2) 
        
        v = v.view(interim_shape).transpose(1, 2) 
        
       
        weight = q @ k.transpose(-1, -2)                                                          
        
        weight /= math.sqrt(self.d_head)                                                          # scaling down 
        weight = F.softmax(weight, dim=-1)
        
       
        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        
        
        output = output.view(input_shape)
        
        
        output = self.out_proj(output)

        
        return output



class res_block(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        """a residual network that accepts image and broadcasts time  """
        
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)     
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)    
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x, time):

        ## accepts an image of dimentsion of (batch , channel ,h ,w ) and time ( batch  , 1280 )  and after linear for time , conv for image
        ## time is broadcast added  to the image and added with the residual image

        res = x
        #image norm -> activate -> convolution 
        x = self.conv_feature(F.silu(self.groupnorm_feature(x)))
        
        #(time norm -> activate -> linear )  & broadcast to  image 
        x = x + (self.linear_time(F.silu(time))).unsqueeze(-1).unsqueeze(-1)
        
        #merged norm -> activate -> convolution 
        x = self.conv_merged(F.silu(self.groupnorm_merged(x)))
        
        
        # residual adding 
        return x + self.residual_layer(res)


class Attentionblock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):

        """a class that accepts an image and a prompt and puts the image through a self and cross attention layers  """
        
        super().__init__()
        
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context): 

        # x: (Batch_Size, channels, h, w)
        # context: (Batch_Size, Seq_Len, Dim) seq-length = 77 (length of prompt)  Dim = dim of the tokens of the prompt  


       

         ###residual self attention layer#
        
        res_long = x                                                            # long  term residual layer instiation of size (Batch, Channels, Height ,Width)   
        
        x = self.conv_input(self.groupnorm(x))                                  # image -> norm -> conv
        b, c, h, w = x.shape                                                    # saving the shape of the initial image input             
        x = (x.view((b, c, h * w))).transpose(-1, -2)                           # reshape (flatten and traspose the channel with (w*h))                                                             
        res_short = x                                                           # short term residual layer instiation
        x = self.attention_1(self.layernorm_1(x))                               # flattened image  -> norm ->self  attention
        x += res_short                                                          # skip connection 


        
        
        ###residual self cross layer#
       
        res_short = x                                                            # short term residual layer instiation of size (Batch_Size, Height * Width, Features)   
        x = self.attention_2(self.layernorm_2(x),context)                        # (flattened image  -> norm ->) + context -> cross attention
        x += res_short                                                           # residual network
        res_short = x                                                            # res layer initiation 
        x = self.layernorm_3(x)                                                  # layer norm 
        
        
        
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)                         # GeGLU implementation   original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
                                                                                  # (Batch , Height * Width , Channels) ->(Batch , Height * Width , 8 * Channels)  
                                                                                  # then using chunk in the last dim to two tensors of  (Batch , Height * Width , Channels*4 )
                                                                                  # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) 
        x = x * F.gelu(gate)                                                      # -> (Batch, Height * Width, Channels * 4) (read later ?)
        
        
        x = self.linear_geglu_2(x)                                                # return the dims of x to (Batch, Height * Width, Channels )
        x += res_short                                                            # residual adding
        x = x.transpose(-1, -2)                                                   # transpose the channel and the (h*w) back 
        x = x.view((b, c, h, w))                                                  # returning the original shape dimension 

        
        return self.conv_output(x) + res_long                                     # adding the long res skip connection


class Sequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, Attentionblock):
                x = layer(x, context)
            elif isinstance(layer, res_block):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')                                                    # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        return self.conv(x)
