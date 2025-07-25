import torch.nn as nn
import torch

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.randn(d_in, d_out),requires_grad=False)
        self.W_key = nn.Parameter(torch.randn(d_in,d_out),requires_grad=False)
        self.W_value= nn.Parameter(torch.randn(d_in,d_out),requires_grad=False)

    def forward (self,x ):
        queries =  x @ self.W_query  # torch.matmul(x, self.W_query)
        keys    =  x @ self.W_key   
        values  =  x @ self.W_value  
 
        attn_score = queries @ keys.T # Attention scores between all queries and keys
        d_k = keys.shape[1]  # Dimension of key vectors 
        attn_score = attn_score / (d_k ** 0.5)  # Scale attention scores
        attn_weights = torch.softmax(attn_score, dim=-1)  # Normalized attention weights
        context = attn_weights @ values 
        return context

class SelfAttestion_v2(nn.Module):
    def __init__(self, d_in, d_out,qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_score = queries @ keys.T
   
        context_length = attn_score.shape[0]
        mask = torch.triu(torch.ones(context_length,context_length), diagonal=1)
        masked = attn_score.masked_fill(mask.bool(),-torch.inf)
        print (masked)

        d_k = keys.shape[1]
        masked = masked / (d_k ** 0.5)
        attn_weights = torch.softmax(masked, dim=1)
        print (attn_weights)





        context = attn_weights @ values
        return context


inputs = torch.tensor([[0.43,0.15,0.89], # Your x1
                       [0.55,0.87,0.66], # journey x2
                       [0.57,0.85,0.64], # starts x3
                       [0.22,0.58,0.33], # with x4
                       [0.77,0.25,0.10], # one x4
                       [0.05,0.80,0.55], # step x6                       
                        ])
d_in = inputs.shape[1]  # Dimension of input vectors
d_out = 2  # Dimension of output vectors
torch.manual_seed(123)
#self_attention_v1 = SelfAttention_v1(d_in,d_out)
#context_v1 = self_attention_v1(inputs)
#print("Context vectors:",context_v1)

self_attention_v2 = SelfAttestion_v2(d_in,d_out)
context_v2 = self_attention_v2(inputs)
#print("Context vectors:",context_v2)





















