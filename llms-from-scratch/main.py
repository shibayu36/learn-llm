import argparse
import time
from pathlib import Path

import torch

from config import Config
from dataset import create_dataloader, join_texts_with_eos, load_texts
from gpt import OneLayerOneHeadGPT
from tokenizer import CharTokenizer
from train import train_model

DATA_DIR = Path("data/fineweb-japanese-10k")


def run_train(config: Config) -> None:
    # 乱数の種を固定する。DataLoader のシャッフル順やモデルの初期値が毎回同じになる
    torch.manual_seed(config.seed)

    train_texts = load_texts(DATA_DIR / "train.jsonl")
    validation_texts = load_texts(DATA_DIR / "validation.jsonl")
    tokenizer = CharTokenizer.from_texts(train_texts)
    print("語彙数:", tokenizer.vocab_size)

    train_ids = join_texts_with_eos(train_texts, tokenizer)
    validation_ids = join_texts_with_eos(validation_texts, tokenizer)

    train_loader = create_dataloader(
        train_ids,
        batch_size=config.batch_size,
        max_length=config.context_length,
        stride=config.context_length,
        shuffle=True,
        drop_last=True,
    )
    # trainと同じtoken列から作るが、shuffle=Falseで先頭から並ぶ。評価を毎回同じ窓で行うため
    train_eval_loader = create_dataloader(
        train_ids,
        batch_size=config.batch_size,
        max_length=config.context_length,
        stride=config.context_length,
        shuffle=False,
        drop_last=False,
    )
    validation_loader = create_dataloader(
        validation_ids,
        batch_size=config.batch_size,
        max_length=config.context_length,
        stride=config.context_length,
        shuffle=False,
        drop_last=False,
    )

    model = OneLayerOneHeadGPT(tokenizer.vocab_size, config)
    # パラメータ数。numel はテンソルの要素数を返す
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    print("パラメータ数:", total_params)

    # AdamWは勾配を受け取ってパラメータを更新する部品。model.parameters() で更新対象の
    # 全パラメータを渡す
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    start_time = time.perf_counter()
    train_model(
        model,
        train_loader,
        train_eval_loader,
        validation_loader,
        optimizer,
        config,
        tokenizer,
        start_context="日本の首都は",
    )
    elapsed = time.perf_counter() - start_time
    print(f"学習時間: {elapsed:.1f}秒")


def main() -> None:
    # argparse はコマンドライン引数を解釈する標準ライブラリ。
    # `uv run main.py train` のように動詞で処理を切り替える
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train")
    args = parser.parse_args()

    if args.command == "train":
        run_train(Config())


if __name__ == "__main__":
    main()
