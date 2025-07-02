import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
inputs = torch.tensor([[0.43,0.15,0.89], # Your x1
                       [0.55,0.87,0.66], # journey x2
                       [0.57,0.85,0.64], # starts x3
                       [0.22,0.58,0.33], # with x4
                       [0.77,0.25,0.10], # one x4
                       [0.05,0.80,0.55], # step x6                       ])
                        ])
# Corresponding words
words=['Your','journey','starts','with','one','step']

# ATTENTION WEIGHT WITH DOT
query = inputs[1] #2nd input
attn_scores_2 =torch.empty(inputs.shape[0])
for i,x_i in enumerate(inputs):
    attn_scores_2[i]= torch.dot(x_i,query) # dot product
print (attn_scores_2)


# ATTENTION WEIGHT NORMALIZED WITH Softmax
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:",attn_weights_2)
print("Sum:",attn_weights_2.sum())

#CONTEXT VECTOR
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2+=x_i*attn_weights_2[i] 
print("Context vector",context_vec_2)

#ALL CONTEXT VECTORS
context_vec = torch.zeros_like(inputs)
for j,q_j in enumerate(inputs):
    attn_scores =torch.empty(inputs.shape[0])
    for i,x_i in enumerate(inputs):
        attn_scores[i]= torch.dot(x_i,q_j) # dot product 
    attn_weights = torch.softmax(attn_scores, dim=0)
    for i,x_i in enumerate(inputs):
        context_vec[j]+=x_i*attn_weights[i] 
print(context_vec)


attn_scores_matrix = torch.empty((inputs.shape[0], inputs.shape[0]))
for j,q_j in enumerate(inputs):
    for i,x_i in enumerate(inputs):
        attn_scores_matrix[j,i]= torch.dot(x_i,q_j) # dot product

# Print the attention scores matrix
print("Attention scores matrix:\n", attn_scores_matrix)
# The attention scores matrix is a square matrix where each row corresponds to a query vector
# and each column corresponds to an input vector.
# Each element (i, j) in the matrix represents the attention score between the i-th input vector and the j-th query vector.
# The attention scores matrix is a square matrix where each row corresponds to a query vector
# and each column corresponds to an input vector.
# Print the shape of the attention scores matrix
print("Shape of attention scores matrix:", attn_scores_matrix.shape)
attn_scores_matrix = inputs @ inputs.T 
print("Attention scores matrix:\n", attn_scores_matrix)    


# Normalize the attention scores to get attention weights
# This will give us a matrix where each row corresponds to the attention weights for each query vector
attn_weights_matrix = torch.softmax(attn_scores_matrix, dim=-1)
print("Attention weights matrix:\n", attn_weights_matrix)
print("All rows sum to 1:", attn_weights_matrix.sum(dim=-1))

context_vec = attn_weights_matrix @ inputs
print("Context vectors:\n", context_vec)
