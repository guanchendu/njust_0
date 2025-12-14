### This is a code for attention
### In the transformer, we can see that there are two type of attention


###########
#1. self attention

import torch
import torch.nn as nn

class selfAttention(nn.Module):
    def __init__(self):
        super().__init__()


        self.embed_dim = 64
        self.num_heads = 8
        self.dropout  = 0.1
        self.attention_layer = nn.MultiheadAttention(embed_dim= self.embed_dim , num_heads= self.num_heads, dropout= self.dropout  )

    def forward(self , x):

        # if x.dim() <= 2:
        #     raise ValueError("Attention : Make sure the dimesion of input Tensor is more than 3")
        # if x.size[1] == 1:
        #     raise ValueError("Attention : Make sure the input is the seq Tensor")
        #
        x,_  = self.attention_layer(x,x,x)
        return x

a = selfAttention()

x = torch.randn(size = (16,3,64))
print(a(x).size())