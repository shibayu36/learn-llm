import torch
from tiny_gpt.config import Config
from tiny_gpt.model import TinyGPT, TransformerBlock
from tiny_gpt.tokenizer import DATA_DIR, Tokenizer, read_texts

tokenizer = Tokenizer.from_texts(read_texts(DATA_DIR / "train.jsonl"))
config = Config()
config.d_model = 4
model = TinyGPT(tokenizer.vocab_size, config)

ids = torch.tensor([tokenizer.encode("日本の")], dtype=torch.long)
positions = torch.arange(ids.shape[1])
embedded = model.token_embedding(ids) + model.position_embedding(positions)

block = TransformerBlock(config.d_model)
print(block(embedded).shape)
print((block(embedded) - embedded)[0])
