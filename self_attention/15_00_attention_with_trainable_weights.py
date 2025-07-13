import torch
inputs = torch.tensor([[0.43,0.15,0.89], # Your x1
                       [0.55,0.87,0.66], # journey x2
                       [0.57,0.85,0.64], # starts x3
                       [0.22,0.58,0.33], # with x4
                       [0.77,0.25,0.10], # one x4
                       [0.05,0.80,0.55], # step x6                       ])
                        ])
# Corresponding words
words=['Your','journey','starts','with','one','step']

x_2 = inputs[1]  # x2
d_in = inputs.shape[1]  # Dimension of input vectors
d_out = 2  # Dimension of output vectors

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.randn(d_in, d_out),requires_grad=False)  # Query weights
W_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)  # Key weights
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
print(W_query)
print(W_key)
print(W_value)  
query_2 = torch.matmul(x_2, W_query)  # Query vector for x2
print("Query vector for x2:", query_2)
key_2 = torch.matmul(x_2, W_key)  # Key vector for x2
print("Key vector for x2:", key_2)          
value_2 = torch.matmul(x_2, W_value)  # Value vector for x2
print("Value vector for x2:", value_2)
queries = torch.matmul(inputs, W_query)  # Query vectors for all inputs
print("Query vectors for all inputs:", queries)   
keys = torch.matmul(inputs, W_key)  # Key vectors for all inputs
print("Key vectors for all inputs:", keys)           
values = torch.matmul(inputs, W_value)  # Value vectors for all inputs
print("Value vectors for all inputs:", values)
print("Shape of queries:", queries.shape)
print("Shape of keys:", keys.shape)     
key_2 = keys[1]  # Key vector for x2
print("Key vector for x2:", key_2)
attn_score_22 = query_2.dot(key_2)  # Attention score between x2 and itself
print("Attention score between x2 and itself:", attn_score_22)
attn_score_2 = query_2 @ keys.T  # Attention scores between x2 and all keys
print("Attention scores between x2 and all keys:", attn_score_2)
attn_score = queries @ keys.T  # Attention scores between all queries and keys
print("Attention scores between all queries and keys:", attn_score)