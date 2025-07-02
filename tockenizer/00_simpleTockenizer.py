import re

class SimpleTokenizer:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str ={i:s for s,i in vocab.items()}

    def encode(self,text):
        #split into token
        preprocessed = re.split(r'([,.?_!;:"()\']|--|\s)',text)
        #removing white space
        preprocessed =[item.strip() for item in preprocessed if item.strip()]
        preprocessed =[
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        #replace spaces before the punctuations
        text =re.sub(r'\s+([,.?_!;:"()\'])',r'\1',text)
        return text



# STEP1 : tokenization
with open("the-verdict.txt","r", encoding="utf-8") as f:
    raw_text =f.read()

#print("Total number of character:", len(raw_text))
#print (raw_text[:99])

#spliting ...
preprocessed = re.split(r'([,.?_!;:"()\']|--|\s)',raw_text)
# Removing white space
preprocessed =[item.strip() for item in preprocessed if item.strip()]
#print(preprocessed[:30])
#print(len(preprocessed))


#STEP 2: creating token IDs
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab_size=len(all_tokens)
print(vocab_size)
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))
for i,item in enumerate(list(vocab.items())[-5:]):
    print(item)
                    


# STEP 3 : encode / decode in SimpleTokenzer class;
tokenizer = SimpleTokenizer(vocab)

# Few more example 
text1= "Hello, do you like tea?"
text2= "In the sunlit terraces of the palace."
text =" <|endoftext|> ".join((text1,text2))
print(text)
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))