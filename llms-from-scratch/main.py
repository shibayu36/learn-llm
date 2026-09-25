import math
from pathlib import Path

import torch

from config import Config
from dataset import create_dataloader, join_texts_with_eos, load_texts
from generate import generate, text_to_token_ids, token_ids_to_text
from gpt import OneLayerOneHeadGPT
from tokenizer import CharTokenizer
from train import calc_loss_batch

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

    # 学習前のモデルで「日本の首都は」の続きを生成する。
    # eval() は評価モードへの切り替え。ドロップアウトなど訓練時だけ働く部品を止める。
    # このモデルにはそういう部品はないが、本にならって生成前に呼ぶ
    model.eval()
    prompt = "日本の首都は"
    prompt_ids = text_to_token_ids(prompt, tokenizer)
    print("promptのID:", prompt_ids.tolist(), "shape:", prompt_ids.shape)
    # temperature=0（greedy）は何回呼んでも同じ文になり、0.8 は呼ぶたびに変わる
    for temperature in [0.0, 0.0, 0.8, 0.8]:
        generated_ids = generate(
            model,
            prompt_ids,
            max_new_tokens=30,
            context_size=config.context_length,
            temperature=temperature,
        )
        print(f"学習前の生成 (T={temperature}):", repr(token_ids_to_text(generated_ids, tokenizer)))

    # 1バッチの損失。本の図5-7の6ステップを手で追って、cross_entropy と同じ値になることを見る
    # 1. ロジット（上で計算済み） 2. softmaxで確率に。[B, T, vocab_size]
    probas = torch.softmax(logits, dim=-1)
    # 3. 各位置で正解の文字に割り当てた確率を取り出す。0番目の系列の先頭3位置だけ表示
    for t in range(3):
        input_char = tokenizer.decode([inputs[0, t].item()])
        target_char = tokenizer.decode([targets[0, t].item()])
        target_proba = probas[0, t, targets[0, t]].item()
        print(f"位置{t} {input_char!r} の次が {target_char!r} である確率: {target_proba:.6f}")
    # 4. 対数を取る 5. 全位置で平均する 6. -1倍する。全位置を素直なループで回す
    total_log_proba = 0.0
    for b in range(inputs.shape[0]):
        for t in range(inputs.shape[1]):
            target_proba = probas[b, t, targets[b, t]]
            total_log_proba += torch.log(target_proba).item()
    manual_loss = -total_log_proba / (inputs.shape[0] * inputs.shape[1])
    loss = calc_loss_batch(inputs, targets, model)
    print("手計算の損失:", manual_loss)
    print("cross_entropyの損失:", loss.item())
    print("ln(vocab_size):", math.log(tokenizer.vocab_size))
    assert abs(manual_loss - loss.item()) < 1e-3
    # パープレキシティ exp(loss) は「次の文字の候補が実質何個に絞れているか」。学習前は語彙数に近い
    print("パープレキシティ:", math.exp(loss.item()))


if __name__ == "__main__":
    main()
