import re
import importlib
import importlib.metadata
import tiktoken

# STEP1 : tokenization
with open("the-verdict.txt","r", encoding="utf-8") as f:
    raw_text =f.read()

#print ("tiktoken version:", importlib.metadata.version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

enc_text= tokenizer.encode(raw_text);
print (len(enc_text)); 

enc_sample = enc_text[50:]

context_size=4 #length  of input
# context size of4 means that the model is trained 
# to look at a sequence of 4 words (or 4 tokens)
# to predict the next word is the sequencd
# the input x is the first 4 tokens [1,2,3,4] and
# the targe y is the next 4 tokens [2,3,4,5

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print (f"x:{x}")
print (f"y:     {y}")

for i in range(1, context_size +1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    #print(context,"----->",desired)
    print(tokenizer.decode(context),"---->",tokenizer.decode([desired]))

