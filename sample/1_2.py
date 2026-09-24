from tiny_gpt.tokenizer import DATA_DIR, Tokenizer, read_texts

tokenizer = Tokenizer.from_texts(read_texts(DATA_DIR / "train.jsonl"))
text = "日本の首都は東京です。"
token_ids = tokenizer.encode(text)
print("語彙数:", tokenizer.vocab_size)
print("入力:", text)
print("token ID:", token_ids)
print("復元:", tokenizer.decode(token_ids))

import torch

ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
x = ids[:-1]
y = ids[1:]
print("元のID:", ids.tolist())
print("入力:", tokenizer.decode(x.tolist()))
print("正解:", tokenizer.decode(y.tolist()))
print(x.shape, y.shape)
assert tokenizer.decode(tokenizer.encode(text)) == text
