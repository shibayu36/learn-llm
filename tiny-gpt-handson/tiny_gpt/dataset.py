import torch

from tiny_gpt.tokenizer import Tokenizer


def encode_documents(texts: list[str], tokenizer: Tokenizer) -> torch.Tensor:
    # 何千もの文書を、文書の区切りにEOSを挟んで1本の長いtoken列につなぐ。
    # 例: ["骨粗しょう症とは", "私たちは呼吸"] という2文書なら
    #   骨 粗 し ょ う 症 と は <eos> 私 た ち は 呼 吸 <eos>
    # という1本の列になる（実際は1文字が1つのIDに置き換わる）。
    # 学習ではこの長い列のどこからでも切り出して使う。
    token_ids: list[int] = []
    for text in texts:
        token_ids.extend(tokenizer.encode(text))
        # 別の文書へ移る境界を、専用のtokenで表す。
        # これがないと「…呼吸<eos>」の続きが次の文書の冒頭に見え、
        # モデルは無関係な文書同士が続いていると学習してしまう。
        token_ids.append(tokenizer.eos_id)
    return torch.tensor(token_ids, dtype=torch.long)
