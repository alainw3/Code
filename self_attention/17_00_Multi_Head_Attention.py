import torch.nn as nn
import torch


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length,
                 dropout,
                 qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self, x):
        b,num_tokens, d_in = x.shape # New batch dimension 
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_score = queries @ keys.transpose(1,2)
   
            
        masked = attn_score.masked_fill(self.mask.bool()[:num_tokens,:num_tokens],-torch.inf)
        #print (masked)

        d_k = keys.shape[-1]
        masked = masked / (d_k ** 0.5)
        attn_weights = torch.softmax(masked, dim=-1)
        #print (attn_weights)

        attn_weights = self.dropout(attn_weights)
        #print(attn_weights)

        context = attn_weights @ values
        return context

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout , num_heads, qkv_bias= False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention( d_in, d_out, context_length, dropout , qkv_bias)
             for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)



inputs = torch.tensor([[0.43,0.15,0.89], # Your x1
                       [0.55,0.87,0.66], # journey x2
                       [0.57,0.85,0.64], # starts x3
                       [0.22,0.58,0.33], # with x4
                       [0.77,0.25,0.10], # one x4
                       [0.05,0.80,0.55], # step x6                       
                        ])

batch = torch.stack((inputs,inputs),dim=0)
print(batch)
context_length = batch.shape[1]
print(context_length)

d_in = inputs.shape[1]  # Dimension of input vectors
d_out = 2  # Dimension of output vectors
torch.manual_seed(123)

num_heads =2 
multiHeadAttentionWrapper = MultiHeadAttentionWrapper(d_in,d_out,context_length,0.0,num_heads)
context_vector = multiHeadAttentionWrapper(batch)
print("Context vectors:",context_vector)
print(context_vector.shape)




















