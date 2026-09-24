import torch
from torch.nn import functional as F
from tiny_gpt.model import FeedForward

mlp = FeedForward(2)
with torch.no_grad():
    mlp.expand.weight[0] = torch.tensor([2.0, 2.0])
    mlp.expand.bias[0] = -3.0

# 各行は [首都の情報, 助詞「は」]
pairs = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
feature = F.gelu(mlp.expand(pairs))[..., 0]
for inputs, value in zip(pairs.tolist(), feature.tolist()):
    print(inputs, f"{value:.2f}")

from tiny_gpt.config import Config
from tiny_gpt.model import TinyGPT
from tiny_gpt.tokenizer import DATA_DIR, Tokenizer, read_texts

tokenizer = Tokenizer.from_texts(read_texts(DATA_DIR / "train.jsonl"))
config = Config()
config.d_model = 4
model = TinyGPT(tokenizer.vocab_size, config)

ids = torch.tensor([tokenizer.encode("日本の")], dtype=torch.long)
positions = torch.arange(ids.shape[1])
embedded = model.token_embedding(ids) + model.position_embedding(positions)

mlp = FeedForward(config.d_model)
changed = embedded.clone()
changed[0, 0] = changed[0, 0] + 10.0

original_output = mlp(embedded)
changed_output = mlp(changed)
torch.testing.assert_close(original_output[0, 2], changed_output[0, 2])
print(original_output.shape)
