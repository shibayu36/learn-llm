import torch

from tiny_gpt.config import Config
from tiny_gpt.generate import generate
from tiny_gpt.model import TinyGPT
from tiny_gpt.tokenizer import DATA_DIR, Tokenizer, read_texts

config = Config()
torch.manual_seed(config.seed)
tokenizer = Tokenizer.from_texts(read_texts(DATA_DIR / "train.jsonl"))
model = TinyGPT(tokenizer.vocab_size, config)
parameter_count = 0
for parameter in model.parameters():
    parameter_count += parameter.numel()

print("学習前のGPT / パラメータ数:", parameter_count)
print(generate(model, tokenizer, "プログラミングを学ぶには、", 80))
