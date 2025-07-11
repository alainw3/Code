import importlib
import importlib.metadata

import tiktoken.load
import tiktoken

print ("tiktoken version:", importlib.metadata.version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea?<|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

integers = tokenizer.encode(text,allowed_special={"<|endoftext|>"})

print (integers)

strings = tokenizer.decode(integers)

print(strings)

integers = tokenizer.encode("Akwirw ier")
print(integers)