from pathlib import Path

import torch

from config import Config
from dataset import create_dataloader, join_texts_with_eos, load_texts
from gpt import OneLayerOneHeadGPT
from tokenizer import CharTokenizer

DATA_DIR = Path("data/fineweb-japanese-10k")


def main() -> None:
    config = Config()
    # 乱数の種を固定する。DataLoader のシャッフル順やモデルの初期値が毎回同じになる
    torch.manual_seed(config.seed)

    train_texts = load_texts(DATA_DIR / "train.jsonl")
    validation_texts = load_texts(DATA_DIR / "validation.jsonl")
    tokenizer = CharTokenizer.from_texts(train_texts)
    print("語彙数:", tokenizer.vocab_size)

    # encode → decode で元に戻ること
    text = "日本の首都は東京です。"
    assert tokenizer.decode(tokenizer.encode(text)) == text

    # 全文書を <|endoftext|> でつないで1本のID列にする
    train_ids = join_texts_with_eos(train_texts, tokenizer)
    validation_ids = join_texts_with_eos(validation_texts, tokenizer)
    print("train token数:", len(train_ids), "validation token数:", len(validation_ids))
    # 1文書目の末尾に <|endoftext|> が入っていること
    first_doc_length = len(tokenizer.encode(train_texts[0]))
    print("1文書目の末尾:", repr(tokenizer.decode(train_ids[first_doc_length - 5:first_doc_length + 2])))

    train_loader = create_dataloader(
        train_ids,
        batch_size=config.batch_size,
        max_length=config.context_length,
        stride=config.context_length,
        shuffle=True,
        drop_last=True,
    )
    validation_loader = create_dataloader(
        validation_ids,
        batch_size=config.batch_size,
        max_length=config.context_length,
        stride=config.context_length,
        shuffle=False,
        drop_last=False,
    )
    print("trainの窓の数:", len(train_loader.dataset), "バッチ数:", len(train_loader))
    print("validationの窓の数:", len(validation_loader.dataset), "バッチ数:", len(validation_loader))

    # 1バッチ取り出す。iter と next はDataLoaderから先頭のバッチを1つ取り出す書き方
    inputs, targets = next(iter(train_loader))
    print("inputs.shape:", inputs.shape, "targets.shape:", targets.shape)
    # 入力と正解が1文字ずれていること。inputs[0] は0番目の系列、[:20] はその先頭20文字
    print("input :", repr(tokenizer.decode(inputs[0][:20].tolist())))
    print("target:", repr(tokenizer.decode(targets[0][:20].tolist())))
    assert torch.equal(inputs[0][1:], targets[0][:-1])

    # モデルに通して [B, T, vocab_size] のロジットが返ること
    model = OneLayerOneHeadGPT(tokenizer.vocab_size, config)
    logits = model(inputs)
    print("logits.shape:", logits.shape)
    assert logits.shape == (config.batch_size, config.context_length, tokenizer.vocab_size)

    # パラメータ数。numel はテンソルの要素数を返す
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    print("パラメータ数:", total_params)


if __name__ == "__main__":
    main()
