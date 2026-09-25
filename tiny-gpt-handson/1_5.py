import torch
from tiny_gpt.config import Config
from tiny_gpt.model import SelfAttention, TinyGPT
from tiny_gpt.tokenizer import DATA_DIR, Tokenizer, read_texts

torch.manual_seed(0)
tokenizer = Tokenizer.from_texts(read_texts(DATA_DIR / "train.jsonl"))

config = Config()
config.d_model = 4
model = TinyGPT(tokenizer.vocab_size, config)


ids = torch.tensor([
    tokenizer.encode("赤い花"),
    tokenizer.encode("青い花"),
], dtype=torch.long)
positions = torch.arange(ids.shape[1])
embedded = model.token_embedding(ids) + model.position_embedding(positions)

attention = SelfAttention(config.d_model)
output, weights = attention(embedded)

print("Attention前の「花」は同じ:", torch.allclose(embedded[0, 2], embedded[1, 2]))
print("Attention後の「花」は同じ:", torch.allclose(output[0, 2], output[1, 2]))
print("「赤い花」の「花」の出力:", output[0, 2].detach())
print("「青い花」の「花」の出力:", output[1, 2].detach())
print("「花」から各文字を参照する割合:", weights[:, 2].detach())
