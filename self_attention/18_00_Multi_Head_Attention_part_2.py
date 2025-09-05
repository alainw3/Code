import torch.nn as nn
import torch



class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length,
                 dropout,
                 num_heads,
                 qkv_bias=False):
        super().__init__()
        assert ( d_out % num_heads ==0),\
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projectinon dim to march  desired output dim


        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

        self.out_proj = nn.Linear(d_out,d_out)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self, x):
        b,num_tokens, d_in = x.shape # New batch dimension 

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a 'num_heads' dimension
        # Unroll last dim : (b, num_tokds, d_out) -> (b, num_tokds, num_heads, head_dim)
        queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)
        keys = keys.view(b,num_tokens,self.num_heads,self.head_dim)
        values = values.view(b,num_tokens,self.num_heads,self.head_dim)

        #Transpose (b,num_tokens,num_heads,head_dim) to (b,num_heads,num_tokens,self.head_dim)
        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        #Attention score 
        attn_score = queries @ keys.transpose(2,3)
   
        #Origial mask trunckated to the number of tho tokens and converted to boolean
        mask_bool =  self.mask.bool()[:num_tokens,:num_tokens]

        #use the mask to fill attenion scores
        attn_score = attn_score.masked_fill(mask_bool,-torch.inf)
            
      

        d_k = keys.shape[-1]
        masked = masked / (d_k ** 0.5)
        attn_weights = torch.softmax(masked, dim=-1)
        attn_weights = self.dropout(attn_weights)
        

        #Shape : (b, num_tokens, num_heads, head_dim)
        context = (attn_weights @ values).transpose(1,2)

        context = context.contiguous().view(b,num_tokens, self.d_out)
        context = self.out_proj(context) # optional projection

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


torch.manual_seed(123)

inputs = torch.tensor([[0.43,0.15,0.89,0.55,0.87,0.66], # row 1
                       [0.57,0.85,0.64,0.22,0.58,0.33], # row 2
                       [0.77,0.25,0.10,0.05,0.80,0.55], # row 3                       
                        ])

batch = torch.stack((inputs,inputs),dim=0)
print(batch.shape)

batch_size, context_length, d_in = batch.shape
d_out = 6  # Dimension of output vectors

num_heads=2
multiHeadAttentionWrapper = MultiHeadAttentionWrapper(d_in,d_out,context_length,0.0,num_heads)
context_vector = multiHeadAttentionWrapper(batch)
print("Context vectors:",context_vector)
print(context_vector.shape)




















