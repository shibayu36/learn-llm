import json
from pathlib import Path

from tokenizer import CharTokenizer

DATA_DIR = Path("data/fineweb-japanese-10k")


def load_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            document = json.loads(line)
            texts.append(document["text"])
    return texts


def main() -> None:
    train_texts = load_texts(DATA_DIR / "train.jsonl")
    tokenizer = CharTokenizer.from_texts(train_texts)
    print("語彙数:", tokenizer.vocab_size)

    # 語彙の先頭・末尾を見る。末尾2つが特別なトークンになっている
    print("語彙の先頭10個:", tokenizer.decode(list(range(10))))
    last_tokens: list[str] = []
    for token_id in range(tokenizer.vocab_size - 4, tokenizer.vocab_size):
        last_tokens.append(tokenizer.int_to_str[token_id])
    print("語彙の末尾4個:", last_tokens)

    # encode → decode で元に戻ること
    text = "日本の首都は東京です。"
    ids = tokenizer.encode(text)
    print("encode:", text, "→", ids)
    print("decode:", ids, "→", tokenizer.decode(ids))
    assert tokenizer.decode(ids) == text

    # 語彙にない文字は <|unk|> になること
    text_with_unknown = "日本の首都は🗼です。"
    ids = tokenizer.encode(text_with_unknown)
    print("未知の文字:", text_with_unknown, "→", tokenizer.decode(ids))
    assert tokenizer.unk_id in ids

    # 保存して読み直しても同じ結果になること
    tmp_path = Path("tmp/vocab.json")
    tmp_path.parent.mkdir(exist_ok=True)
    tokenizer.save(tmp_path)
    loaded = CharTokenizer.load(tmp_path)
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.vocab_size == tokenizer.vocab_size

    # train / validation の token 数
    validation_texts = load_texts(DATA_DIR / "validation.jsonl")
    for name, texts in (("train", train_texts), ("validation", validation_texts)):
        total_tokens = 0
        unknown_tokens = 0
        for text in texts:
            for token_id in tokenizer.encode(text):
                total_tokens += 1
                if token_id == tokenizer.unk_id:
                    unknown_tokens += 1
        print(name, "token数:", total_tokens, "<|unk|>の数:", unknown_tokens)


if __name__ == "__main__":
    main()
