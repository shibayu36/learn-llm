import json
from pathlib import Path

# 本の SimpleTokenizerV2 と同じ2つの特別なトークン。
# <|endoftext|> は無関係な文書どうしの区切り、<|unk|> は語彙にない文字の置き換え
END_OF_TEXT = "<|endoftext|>"
UNKNOWN = "<|unk|>"


# 本の SimpleTokenizerV2 の「単語」を「1文字」に置き換えたトークナイザ。
# 語彙は「文字 → ID」の表で、encodeは文字列をIDの列に、decodeはその逆を行う
class CharTokenizer:
    def __init__(self, vocab: list[str]) -> None:
        # vocab はトークン文字列のリストで、リストの添字がそのままトークンIDになる。
        # 末尾2つは <|endoftext|> と <|unk|>
        self.int_to_str: dict[int, str] = {}
        self.str_to_int: dict[str, int] = {}
        for token_id in range(len(vocab)):
            token = vocab[token_id]
            self.int_to_str[token_id] = token
            self.str_to_int[token] = token_id
        self.eos_id: int = self.str_to_int[END_OF_TEXT]
        self.unk_id: int = self.str_to_int[UNKNOWN]

    @property
    def vocab_size(self) -> int:
        return len(self.int_to_str)

    def encode(self, text: str) -> list[int]:
        # Pythonの文字列をforで回すと1文字ずつ取り出せる。語彙にない文字は <|unk|> のID
        ids: list[int] = []
        for char in text:
            if char in self.str_to_int:
                ids.append(self.str_to_int[char])
            else:
                ids.append(self.unk_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        text = ""
        for token_id in ids:
            text += self.int_to_str[token_id]
        return text

    # 訓練テキストに現れる文字を集めて語彙を作る。文字コード順に並べるので、
    # 同じテキストからは必ず同じIDの割り当てになる
    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharTokenizer":
        unique_chars: set[str] = set()
        for text in texts:
            for char in text:
                unique_chars.add(char)
        vocab = sorted(unique_chars)
        vocab.append(END_OF_TEXT)
        vocab.append(UNKNOWN)
        return cls(vocab)

    # 語彙をJSONで保存・復元する。学習済みモデルはこの語彙のIDを前提にしているので、
    # モデルと語彙は必ず組で保存する
    def save(self, path: Path) -> None:
        vocab: list[str] = []
        for token_id in range(self.vocab_size):
            vocab.append(self.int_to_str[token_id])
        path.write_text(
            json.dumps(vocab, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        vocab = json.loads(path.read_text(encoding="utf-8"))
        return cls(vocab)
