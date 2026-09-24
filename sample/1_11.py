import torch
from torch.nn import functional as F

target = torch.tensor([1], dtype=torch.long)
uncertain = torch.tensor([[0.0, 0.0, 0.0]])
confident = torch.tensor([[0.0, 3.0, 0.0]])
print(F.cross_entropy(uncertain, target).item())
print(F.cross_entropy(confident, target).item())

logits = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
loss = F.cross_entropy(logits, target)
loss.backward()
print(logits.grad)

from tiny_gpt.config import Config
from tiny_gpt.dataset import encode_documents, load_data, make_batch
from tiny_gpt.model import TinyGPT
from tiny_gpt.tokenizer import DATA_DIR, Tokenizer, read_texts
from tiny_gpt.train import batch_loss, train_model

tokenizer = Tokenizer.from_texts(read_texts(DATA_DIR / "train.jsonl"))
config = Config()
config.steps = 100
config.eval_every = 50
torch.manual_seed(config.seed)

joined = encode_documents(["日本の首都は東京です。", "猫は"], tokenizer)
print("つないだID:", joined.tolist())
print("復元:", tokenizer.decode(joined.tolist()))

train_ids, validation_ids = load_data(tokenizer)
print("trainのtoken数:", len(train_ids))
model = TinyGPT(tokenizer.vocab_size, config)

generator = torch.Generator()
generator.manual_seed(config.seed)
x, y = make_batch(train_ids, config, generator)
print(x.shape, y.shape)
print("入力:", tokenizer.decode(x[0, :16].tolist()))
print("正解:", tokenizer.decode(y[0, :16].tolist()))
print("学習前のloss:", round(batch_loss(model, x, y).item(), 4))

history, elapsed = train_model(model, train_ids, validation_ids, config)
print("秒数:", round(elapsed, 1))
