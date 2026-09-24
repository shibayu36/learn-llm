import json
from pathlib import Path


DATA_DIR = Path("data/fineweb-japanese-10k")
# 文字とは別に語彙の先頭へ予約する特別なtoken。文字列は表示用の名前で、
# 文章の中には現れない。
# EOS: 文書の終わり。文書の末尾にだけ付ける。GPTはこれも予測対象として学習し、
#      生成時に選んだら文章を終了する。
# UNK: 語彙にない文字。trainに出なかった文字がvalidationやpromptに現れたとき、
#      その文字だけこのtokenに置き換える。
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


def read_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            document = json.loads(line)
            texts.append(document["text"])
    return texts


class Tokenizer:
    def __init__(self, tokens: list[str]) -> None:
        # ID順に並べたtokenの表。token IDはこの表の添字。
        self.tokens: list[str] = tokens
        self.token_to_id: dict[str, int] = {}
        for token_id, token in enumerate(tokens):
            self.token_to_id[token] = token_id
        self.vocab_size: int = len(tokens)
        self.eos_id: int = self.token_to_id[EOS_TOKEN]
        self.unk_id: int = self.token_to_id[UNK_TOKEN]

    @classmethod
    def from_texts(cls, texts: list[str]) -> "Tokenizer":
        chars: set[str] = set()
        for text in texts:
            chars.update(text)
        return cls([EOS_TOKEN, UNK_TOKEN] + sorted(chars))

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.tokens, ensure_ascii=False), encoding="utf-8")

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for char in text:
            token_ids.append(self.token_to_id.get(char, self.unk_id))
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        chars: list[str] = []
        for token_id in token_ids:
            chars.append(self.tokens[token_id])
        return "".join(chars)
