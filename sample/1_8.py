import torch
from tiny_gpt.config import Config
from tiny_gpt.model import LayerNorm, TinyGPT
from tiny_gpt.tokenizer import DATA_DIR, Tokenizer, read_texts

tokenizer = Tokenizer.from_texts(read_texts(DATA_DIR / "train.jsonl"))
config = Config()
config.d_model = 4
model = TinyGPT(tokenizer.vocab_size, config)

ids = torch.tensor([tokenizer.encode("日本の")], dtype=torch.long)
positions = torch.arange(ids.shape[1])
embedded = model.token_embedding(ids) + model.position_embedding(positions)

norm = LayerNorm(config.d_model)
wide = embedded * 10 + 5
normalized = norm(wide)
print(wide[0].mean(dim=-1), wide[0].var(dim=-1, unbiased=False))
print(normalized[0].mean(dim=-1), normalized[0].var(dim=-1, unbiased=False))
torch.testing.assert_close(norm(wide), norm(wide * 10))
