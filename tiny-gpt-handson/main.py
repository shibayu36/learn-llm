import torch
from tiny_gpt.config import Config
from tiny_gpt.model import TinyGPT
from tiny_gpt.tokenizer import DATA_DIR, Tokenizer, read_texts

tokenizer = Tokenizer.from_texts(read_texts(DATA_DIR / "train.jsonl"))

text = "日本の首都は東京です。"
ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
x = ids[:-1]
y = ids[1:]
print("元のID:", ids.tolist())
print("入力:", tokenizer.decode(x.tolist()))
print("正解:", tokenizer.decode(y.tolist()))
print(x.shape, y.shape)
assert tokenizer.decode(tokenizer.encode(text)) == text

config = Config()
config.d_model = 4
model = TinyGPT(tokenizer.vocab_size, config)

ids = torch.tensor([tokenizer.encode("日本の")], dtype=torch.long)
print("入力:", tokenizer.decode(ids[0].tolist()))
x = model(ids)
print(x.shape)
print(x[0])

# 1.4
print(ids.shape)
positions = torch.arange(ids.shape[1])
print(model.position_embedding.weight[positions])
embedded = (
    model.token_embedding.weight[ids] + model.position_embedding.weight[positions]
)
print(torch.equal(x, embedded))
