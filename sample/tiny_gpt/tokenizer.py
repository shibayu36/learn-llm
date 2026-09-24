import json
from pathlib import Path

from tokenizers import Tokenizer as BpeTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers


DATA_DIR = Path("data/fineweb-japanese-10k")
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
EOS_TOKEN = "<|endoftext|>"
VOCAB_SIZE = 8192


def read_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            document = json.loads(line)
            texts.append(document["text"])
    return texts


def train_tokenizer(texts: list[str], vocab_size: int) -> BpeTokenizer:
    tokenizer = BpeTokenizer(models.BPE())
    # 256種類のbyteを出発点にするので、学習にない文字もID列へ変換できる。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[EOS_TOKEN],
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    return tokenizer


class Tokenizer:
    def __init__(self, path: Path = TOKENIZER_PATH) -> None:
        self.backend: BpeTokenizer = BpeTokenizer.from_file(str(path))
        self.vocab_size: int = self.backend.get_vocab_size()
        eos_id = self.backend.token_to_id(EOS_TOKEN)
        if eos_id is None:
            raise ValueError("文書終端tokenがありません")
        self.eos_id: int = eos_id

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.backend.decode(token_ids)

    def save(self, path: Path) -> None:
        self.backend.save(str(path))


def main() -> None:
    if not TOKENIZER_PATH.exists():
        # validationの内容を使わず、trainだけから分割規則と語彙を作る。
        texts = read_texts(DATA_DIR / "train.jsonl")
        backend = train_tokenizer(texts, VOCAB_SIZE)
        backend.save(str(TOKENIZER_PATH))
    tokenizer = Tokenizer()
    text = "日本の首都は東京です。"
    token_ids = tokenizer.encode(text)
    print("語彙数:", tokenizer.vocab_size)
    print("入力:", text)
    print("token ID:", token_ids)
    print("復元:", tokenizer.decode(token_ids))


if __name__ == "__main__":
    main()
