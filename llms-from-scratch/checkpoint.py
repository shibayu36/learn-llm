import json
from pathlib import Path

import torch
import torch.nn as nn

from config import Config
from gpt import OneLayerOneHeadGPT, OneHeadGPT, GPT, JpCharGPT2
from tokenizer import CharTokenizer, JpCharGPT2Tokenizer

# 本の 5.4「モデルの重みの保存と読み込み」に相当する。学習済みのモデル・設定・語彙を
# 1つのディレクトリにまとめて保存し、学習をやり直さずに読み込めるようにする


def save_run(run_dir: Path, model: nn.Module, config: Config, tokenizer: CharTokenizer) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # state_dict() は「パラメータ名 → テンソル」の辞書。torch.save はそれをファイルに書く
    torch.save(model.state_dict(), run_dir / "model.pt")

    # type(model).__name__ はモデルのクラス名を文字列にしたもの（例: "OneLayerOneHeadGPT"）。
    # 読み込み時にどのクラスを組み立てるか決めるために、クラス名も保存する
    data = config.to_dict()
    data["model"] = type(model).__name__
    (run_dir / "config.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # モデルは学習時の語彙のIDを前提にしているので、語彙もモデルと組で保存する
    tokenizer.save(run_dir / "vocab.json")


def load_run(run_dir: Path) -> tuple[nn.Module, Config, CharTokenizer]:
    data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config = Config.from_dict(data)

    tokenizer = CharTokenizer.load(run_dir / "vocab.json")
    # 学習済みの日本語GPT-2だけencodeの規則が違う
    if data["model"] == "JpCharGPT2":
        tokenizer = JpCharGPT2Tokenizer.load(run_dir / "vocab.json")

    if data["model"] == "OneLayerOneHeadGPT":
        model = OneLayerOneHeadGPT(tokenizer.vocab_size, config)
    elif data["model"] == "OneHeadGPT":
        model = OneHeadGPT(tokenizer.vocab_size, config)
    elif data["model"] == "GPT":
        model = GPT(tokenizer.vocab_size, config)
    elif data["model"] == "JpCharGPT2":
        model = JpCharGPT2(tokenizer.vocab_size, config)
    else:
        raise ValueError(f"未対応のモデル: {data['model']}")

    # load_state_dict は、同じ構造のモデルに保存しておいた辞書の値を流し込む
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    return model, config, tokenizer
