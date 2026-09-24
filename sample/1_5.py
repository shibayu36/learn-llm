import torch
from tiny_gpt.config import Config
from tiny_gpt.model import SelfAttention, TinyGPT, scaled_attention
from tiny_gpt.tokenizer import DATA_DIR, Tokenizer, read_texts

tokenizer = Tokenizer.from_texts(read_texts(DATA_DIR / "train.jsonl"))

config = Config()
config.d_model = 4
model = TinyGPT(tokenizer.vocab_size, config)

ids = torch.tensor([tokenizer.encode("日本の")], dtype=torch.long)
positions = torch.arange(ids.shape[1])
embedded = model.token_embedding(ids) + model.position_embedding(positions)

attention = SelfAttention(config.d_model)
q = attention.query(embedded)
k = attention.key(embedded)
v = attention.value(embedded)
mixed, weights = scaled_attention(q, k, v, causal=False)
output = attention.output(mixed)
print(output.shape, weights.shape)
print(weights[0])

q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
v = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])
output, weights = scaled_attention(q, k, v, causal=False)

print(weights)
print(output)
torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 2))
