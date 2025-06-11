import re
import importlib
import importlib.metadata
import tiktoken
import torch

from torch.utils.data import Dataset, DataLoader


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


def create_data_loader_v1(txt,batch_size=4, max_length=256,
                          stride=128, shuffle= True, drop_last = True,
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

print("Pytorch version:", torch.__version__)
dataloader = create_data_loader_v1(raw_text,
                                   batch_size=8,
                                   max_length=4,
                                   stride=4,
                                   shuffle=False)

data_iter= iter(dataloader)
#first_batch = next(data_iter)
#print(first_batch)
#second_batch = next(data_iter)
#print(second_batch)


inputs, targets = next(data_iter)
print("Input:\n",inputs)
print("\nTargets:\n",targets)
