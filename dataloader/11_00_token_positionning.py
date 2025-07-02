import torch 
import tiktoken
from torch.utils.data import Dataset, DataLoader



vocab_size = 50257
output_dim = 256

torch.manual_seed(123)
token_embedding_layer  = torch.nn.Embedding(vocab_size,output_dim)


class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,max_length, stride):
        self.input_ids  =[]
        self.target_ids =[]

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<!endoftext|>"})

        #Use a sliding windows to chunk he book int overlapping sequences 
        # of max_lengh
        for  i in range(0,len(token_ids)- max_length,stride):
            input_chunk     = token_ids[i   :i  +   max_length     ]
            target_chunk    = token_ids[i+1 :i  +   max_length+ 1  ]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]


max_length = 4

def create_data_loader_v1(txt,batch_size=8, max_length=max_length,
                          stride=max_length, shuffle= True, drop_last = True,
                          num_workers=0):
    
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    #create datas
    dataset = GPTDatasetV1(txt,tokenizer, max_length,stride)

    #create dataloader  
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle = shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# STEP1 : tokenization
with open("the-verdict.txt","r", encoding="utf-8") as f:
    raw_text =f.read()

dataloader = create_data_loader_v1(raw_text,
                                   batch_size=8,
                                   max_length=max_length,
                                   stride=max_length,
                                   shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print ("Token IDs:\n", inputs)
print("\nInputs shape:\n",inputs.shape) 

token_embeddings = token_embedding_layer (inputs)
print("\nToken embeddings shape:\n",token_embeddings.shape)
#print(token_embedding_layer(torch.tensor([40])).shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length,output_dim)
pos_embeddings  = pos_embedding_layer(torch.arange(max_length))
print("\nPos embeddings shape:\n",pos_embeddings.shape)
#print(pos_embedding_layer(torch.tensor([1])))

input_embeddings = token_embeddings + pos_embeddings
print("\nInput embeddings shape:\n",input_embeddings.shape)



