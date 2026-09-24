import torch
from tiny_gpt.config import Config
from tiny_gpt.model import TinyGPT
from tiny_gpt.tokenizer import DATA_DIR, Tokenizer, read_texts

tokenizer = Tokenizer.from_texts(read_texts(DATA_DIR / "train.jsonl"))

config = Config()
config.d_model = 4
model = TinyGPT(tokenizer.vocab_size, config)

ids = torch.tensor([tokenizer.encode("東京と京都")], dtype=torch.long)
token_vectors = model.token_embedding(ids)
positions = torch.arange(ids.shape[1])
x = token_vectors + model.position_embedding(positions)

print("位置を足す前")
print("位置1の「京」:", token_vectors[0, 1].detach())
print("位置3の「京」:", token_vectors[0, 3].detach())
print("位置を足した後")
print("位置1の「京」:", x[0, 1].detach())
print("位置3の「京」:", x[0, 3].detach())
