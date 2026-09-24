import torch

from tiny_gpt.config import Config
from tiny_gpt.tokenizer import DATA_DIR, Tokenizer, read_texts


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


def load_data(tokenizer: Tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    train_texts = read_texts(DATA_DIR / "train.jsonl")
    validation_texts = read_texts(DATA_DIR / "validation.jsonl")
    train_ids = encode_documents(train_texts, tokenizer)
    validation_ids = encode_documents(validation_texts, tokenizer)
    return train_ids, validation_ids


def make_batch(
    data: torch.Tensor,
    config: Config,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 長い1本のtoken列から、ランダムな位置でT個ずつ切り出した「窓」をB本集める。
    # 文書の先頭から順に読むのではなく、毎回ばらばらの場所から練習問題を作る。
    # 例（T=8）: ある窓が「は、上記の問題へ」なら、
    #   入力 x: は 、 上 記 の 問 題 へ
    #   正解 y: 、 上 記 の 問 題 へ の
    # となり、「は→、」「は、→上」「は、上→記」… と
    # 「ここまでを読んで次の1文字を当てる」問題がT個ぶん一度にできる。
    starts = torch.randint(
        low=0,
        high=len(data) - config.context_length,
        size=(config.batch_size,),
        generator=generator,
    )
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for start_tensor in starts:
        start = int(start_tensor.item())
        end = start + config.context_length
        # 各位置の正解は1token先。入力と正解の長さはどちらもTに揃える。
        inputs.append(data[start:end])
        targets.append(data[start + 1:end + 1])

    # B本の窓を縦に並べて、shape [B, T] の入力と正解にする。
    x = torch.stack(inputs)
    y = torch.stack(targets)
    return x, y
