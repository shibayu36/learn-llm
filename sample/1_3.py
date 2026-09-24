import torch
from tiny_gpt.config import Config
from tiny_gpt.model import TinyGPT
from tiny_gpt.tokenizer import DATA_DIR, Tokenizer, read_texts

tokenizer = Tokenizer.from_texts(read_texts(DATA_DIR / "train.jsonl"))

config = Config()
config.d_model = 4
model = TinyGPT(tokenizer.vocab_size, config)

ids = torch.tensor([tokenizer.encode("日本の")], dtype=torch.long)
print("入力:", tokenizer.decode(ids[0].tolist()))
print("ID:", ids.tolist())
x = model.token_embedding(ids)
print(x.shape)
print(x[0])
print(torch.equal(x[0], model.token_embedding.weight[ids[0]]))
