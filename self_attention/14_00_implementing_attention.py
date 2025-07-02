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

# Extract x,y,z coordinates
x_coords = inputs[:,0].numpy()
y_coords = inputs[:,1].numpy()
z_coords = inputs[:,2].numpy()

# create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')


# Define a list of colors for the vectors
color =['r','g','b','c','m','y']

# Plot each  vecto
for x,y,z ,word, color in zip (x_coords,y_coords,z_coords,words, color):
    ax.scatter(x,y,z)
    ax.quiver(0,0,0, x,y,z, color=color)
    ax.text(x,y,z, word, fontsize=10, color = color)

# Set labels for axes
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Set plot limit to keep arros within the plot boudaries
ax.set_xlim([0,1])
ax.set_ylim([0,1])


#plt.title('3D Plot of Word Embeddings')
#plt.show()

# ATTENTION WEIGHT Temp
query = inputs[1] #2nd input
attn_scores_2 =torch.empty(inputs.shape[0])
for i,x_i in enumerate(inputs):
    attn_scores_2[i]= torch.dot(x_i,query) # dot product
print (attn_scores_2)

# ATTENTION WEIGHT Naive
attn_weights_2_temp =    attn_scores_2 /  attn_scores_2.sum()
print  ("Attention weights :" ,attn_weights_2_temp)
print ("Sum :",attn_weights_2_temp.sum())
print(attn_weights_2_temp.norm())
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:",attn_weights_2_naive)
print("Sum:",attn_weights_2_naive.sum())


# ATTENTION WEIGHT Softmax
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:",attn_weights_2)
print("Sum:",attn_weights_2.sum())