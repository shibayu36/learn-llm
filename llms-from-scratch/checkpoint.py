import json
from pathlib import Path

import torch
import torch.nn as nn

from config import Config
from gpt import OneLayerOneHeadGPT
from tokenizer import CharTokenizer

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

    if data["model"] == "OneLayerOneHeadGPT":
        model = OneLayerOneHeadGPT(tokenizer.vocab_size, config)
    else:
        raise ValueError(f"未対応のモデル: {data['model']}")

    # load_state_dict は、同じ構造のモデルに保存しておいた辞書の値を流し込む
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    return model, config, tokenizer
