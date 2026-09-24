import torch
from tiny_gpt.config import Config
from tiny_gpt.model import TinyGPT
from tiny_gpt.tokenizer import DATA_DIR, Tokenizer, read_texts

tokenizer = Tokenizer.from_texts(read_texts(DATA_DIR / "train.jsonl"))
torch.manual_seed(42)
config = Config()
model = TinyGPT(tokenizer.vocab_size, config)
model.eval()

first_ids = tokenizer.encode("日本の首都は東京です。")
second_ids = tokenizer.encode("日本の通貨は円です。")
first = torch.tensor([first_ids], dtype=torch.long)
second = torch.tensor([second_ids], dtype=torch.long)

with torch.no_grad():
    first_logits = model(first)
    second_logits = model(second)

print(first_logits.shape)
torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3])

parameter_count = 0
for parameter in model.parameters():
    parameter_count += parameter.numel()
print("パラメータ数:", parameter_count)
