import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
# print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("words count:", len(preprocessed))
# print(preprocessed[:30])

# class SimpleTokenizerV1:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i:s for s,i in vocab.items()}

#     def encode(self, text):
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

#         preprocessed = [
#             item.strip() for item in preprocessed if item.strip()
#         ]
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids

#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])
#         # Replace spaces before the specified punctuations
#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
#         return text

# tokenizer = SimpleTokenizerV1(vocab)
# text = """"It's the last he painted, you know,"
#            Mrs. Gisburn said with pardonable pride."""
# ids = tokenizer.encode(text)
# print(ids)
# decoded = tokenizer.decode(ids)
# print(decoded)


# v2
# all_tokens = sorted(list(set(preprocessed)))
# all_tokens.extend(["<|endoftext|>", "<|unk|>"])
# vocab = {token:integer for integer,token in enumerate(all_tokens)}
# print(len(vocab.items()))

# class SimpleTokenizerV2:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = { i:s for s,i in vocab.items()}
    
#     def encode(self, text):
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#         preprocessed = [item.strip() for item in preprocessed if item.strip()]
#         preprocessed = [
#             item if item in self.str_to_int 
#             else "<|unk|>" for item in preprocessed
#         ]

#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
        
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])
#         # Replace spaces before the specified punctuations
#         text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
#         return text

# tokenizer = SimpleTokenizerV2(vocab)

# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."

# text = " <|endoftext|> ".join((text1, text2))

# print(text)
# print(tokenizer.encode(text))
# print(tokenizer.decode(tokenizer.encode(text)))


tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)

enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4

dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# トークン埋め込み
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# 位置埋め込み
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
