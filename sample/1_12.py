import torch

logits = torch.tensor([2.0, 1.0, 0.0])
for temperature in (0.5, 1.0, 2.0):
    probabilities = torch.softmax(logits / temperature, dim=-1)
    print(temperature, probabilities.tolist())

from tiny_gpt.config import Config
from tiny_gpt.generate import generate
from tiny_gpt.model import TinyGPT
from tiny_gpt.tokenizer import DATA_DIR, Tokenizer, read_texts

tokenizer = Tokenizer.from_texts(read_texts(DATA_DIR / "train.jsonl"))
config = Config()
torch.manual_seed(config.seed)
untrained = TinyGPT(tokenizer.vocab_size, config)
print("sampling:", generate(untrained, tokenizer, "日本の首都は", 20))
print("greedy:", generate(untrained, tokenizer, "日本の首都は", 20, method="greedy"))
