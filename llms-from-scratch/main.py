import argparse
import json
import time
from pathlib import Path

import torch

from checkpoint import load_run, save_run
from config import Config
from dataset import create_dataloader, join_texts_with_eos, load_texts
from evaluate import MAX_NEW_TOKENS, evaluate_validation, generate_samples
from generate import generate, text_to_token_ids, token_ids_to_text
from gpt import OneLayerOneHeadGPT
from plot import save_loss_plot
from tokenizer import CharTokenizer
from train import train_model

DATA_DIR = Path("data/fineweb-japanese-10k")
PROMPTS_PATH = Path(__file__).parent / "data/evaluation-prompts.json"
RUNS_DIR = Path("runs")


def run_train(config: Config, run_name: str) -> None:
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
    steps, train_losses, validation_losses = train_model(
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

    run_dir = RUNS_DIR / run_name
    save_run(run_dir, model, config, tokenizer)
    training = {
        "run_name": run_name,
        "config": config.to_dict(),
        "steps": steps,
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "elapsed_seconds": elapsed,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps({"training": training}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_loss_plot(training, run_dir / "loss.png")
    print("保存先:", run_dir)


def run_generate(run_name: str, prompt: str, temperature: float, max_new_tokens: int) -> None:
    model, config, tokenizer = load_run(RUNS_DIR / run_name)
    model.eval()
    context_ids = text_to_token_ids(prompt, tokenizer)
    token_ids = generate(
        model,
        context_ids,
        max_new_tokens=max_new_tokens,
        context_size=config.context_length,
        temperature=temperature,
    )
    print(token_ids_to_text(token_ids, tokenizer))


def run_evaluate(run_name: str) -> None:
    start_time = time.perf_counter()
    run_dir = RUNS_DIR / run_name
    model, config, tokenizer = load_run(run_dir)
    texts = load_texts(DATA_DIR / "validation.jsonl")
    token_ids = join_texts_with_eos(texts, tokenizer)
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))

    validation = evaluate_validation(model, token_ids, config.context_length)
    samples = generate_samples(
        model, tokenizer, config.context_length, prompts,
        temperature=0.0, seeds=[None],
    )
    sampling_prompts: list[dict] = []
    for entry in prompts:
        if entry.get("sampling", False):
            sampling_prompts.append(entry)
    samples.extend(generate_samples(
        model, tokenizer, config.context_length, sampling_prompts,
        temperature=0.8, seeds=list(range(42, 52)),
    ))
    elapsed = time.perf_counter() - start_time

    parameter_count = 0
    for parameter in model.parameters():
        parameter_count += parameter.numel()

    metrics_path = run_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["evaluation"] = {
        "parameter_count": parameter_count,
        "validation": validation,
        "generation": {"max_new_tokens": MAX_NEW_TOKENS, "samples": samples},
        "elapsed_seconds": elapsed,
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"validation loss（全バッチ）: {validation['loss']:.6f}")
    print(f"評価token数: {validation['evaluated_tokens']}")
    print(f"生成サンプル数: {len(samples)}")
    print(f"評価時間: {elapsed:.1f}秒")
    print("保存先:", metrics_path)


def main() -> None:
    # argparse はコマンドライン引数を解釈する標準ライブラリ。
    # `uv run main.py train` のように動詞で処理を切り替える
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--run-name", default="l1h1")

    # promptは`--`を付けない位置引数。コマンドラインで順番どおりに書けば渡せる。
    # --run-nameなどのオプション引数は`--名前 値`の形で、書く順番は自由で省略もできる
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("prompt")
    generate_parser.add_argument("--run-name", default="l1h1")
    generate_parser.add_argument("--temperature", type=float, default=0.0)
    generate_parser.add_argument("--max-new-tokens", type=int, default=50)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--run-name", default="l1h1")

    args = parser.parse_args()

    if args.command == "train":
        run_train(Config(), args.run_name)
    elif args.command == "generate":
        run_generate(args.run_name, args.prompt, args.temperature, args.max_new_tokens)
    elif args.command == "evaluate":
        run_evaluate(args.run_name)


if __name__ == "__main__":
    main()
