import argparse
import json
from pathlib import Path

import huggingface_hub
import torch
from safetensors.torch import load_file
from tokenizers import ByteLevelBPETokenizer

from checkpoint import save_run
from config import Config
from gpt import JpCharGPT2
from tokenizer import END_OF_TEXT, UNKNOWN, JpCharGPT2Tokenizer

# 本の 5.5「OpenAI から事前学習済みの重みを読み込む」に相当する。Hugging Faceで公開されている
# GPT-2の重みと語彙を、自作の JpCharGPT2 と JpCharGPT2Tokenizer の形に変換して runs/ に保存する。
# 保存したrunは、自作モデルと同じ generate・evaluate・visualize-* で使える

DEFAULT_MODEL_ID = "ku-nlp/gpt2-small-japanese-char"
DEFAULT_RUN_NAME = "ku-nlp-gpt2-small-char"
RUNS_DIR = Path("runs")
FILENAMES = ["model.safetensors", "config.json", "vocab.json", "merges.txt"]


def download_files(model_id: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for filename in FILENAMES:
        paths[filename] = Path(huggingface_hub.hf_hub_download(model_id, filename))
    return paths


def build_config(hf_config: dict) -> Config:
    # batch_size・steps・learning_rate など学習用の項目は Config() の既定のまま
    config = Config()
    config.n_layers = hf_config["n_layer"]
    config.d_model = hf_config["n_embd"]
    config.n_heads = hf_config["n_head"]
    config.context_length = hf_config["n_positions"]
    return config


def convert_state_dict(
    hf_state: dict[str, torch.Tensor], config: Config
) -> dict[str, torch.Tensor]:
    # 公開されている重みのキーは "transformer.h.0.attn..." のように先頭に transformer. が
    # 付くものと付かないものがあるので、外して同じ名前で扱う。
    # HF側の h.{i}.attn.bias・attn.masked_bias は Causal mask の固定の表、lm_head.weight
    # は wte.weight と同じ行列なので、あっても使わない
    state: dict[str, torch.Tensor] = {}
    for key in hf_state:
        name = key
        if name.startswith("transformer."):
            name = name[len("transformer."):]
        state[name] = hf_state[key]

    d_model = config.d_model
    head_dim = d_model // config.n_heads
    result: dict[str, torch.Tensor] = {}

    # GPT-2は出力ヘッドにトークン埋め込みと同じ行列を使う（重み共有）。
    # state_dict には out_head.weight の名前でも出てくる。load_state_dict(strict=True) は
    # 両方の名前を要求するので、同じテンソルを両方に入れる
    result["tok_emb.weight"] = state["wte.weight"]
    result["out_head.weight"] = state["wte.weight"]
    result["pos_emb.weight"] = state["wpe.weight"]
    result["final_norm.scale"] = state["ln_f.weight"]
    result["final_norm.shift"] = state["ln_f.bias"]

    for i in range(config.n_layers):
        hf_prefix = f"h.{i}."
        my_prefix = f"trf_blocks.{i}."

        result[my_prefix + "norm1.scale"] = state[hf_prefix + "ln_1.weight"]
        result[my_prefix + "norm1.shift"] = state[hf_prefix + "ln_1.bias"]
        result[my_prefix + "norm2.scale"] = state[hf_prefix + "ln_2.weight"]
        result[my_prefix + "norm2.shift"] = state[hf_prefix + "ln_2.bias"]

        # GPT-2は全ヘッドのQ・K・Vを1つの行列 c_attn にまとめて持っている。
        # 列がQ・K・Vの順に3等分され、それぞれの中でヘッド0、ヘッド1、…の順に
        # head_dim 列ずつ並ぶ。D=768・12ヘッドなら head_dim=64 で、Qのヘッド0は列0〜63、
        # Qのヘッド1は列64〜127、Kのヘッド0は列768〜831。
        # 自作モデルはヘッドごとに別の W_query・W_key・W_value を持つので、ここで
        # ヘッドごとに切り分ける。
        # HFのConv1Dは重みを [入力, 出力] で持ち、nn.Linear は [出力, 入力] で持つので、
        # 切り出した重みは転置して渡す
        c_attn_weight = state[hf_prefix + "attn.c_attn.weight"]
        c_attn_bias = state[hf_prefix + "attn.c_attn.bias"]
        projections = ["W_query", "W_key", "W_value"]
        for p in range(3):
            for h in range(config.n_heads):
                start = p * d_model + h * head_dim
                end = start + head_dim
                head_prefix = f"{my_prefix}att.heads.{h}.{projections[p]}."
                result[head_prefix + "weight"] = c_attn_weight[:, start:end].T
                result[head_prefix + "bias"] = c_attn_bias[start:end]

        result[my_prefix + "att.out_proj.weight"] = state[hf_prefix + "attn.c_proj.weight"].T
        result[my_prefix + "att.out_proj.bias"] = state[hf_prefix + "attn.c_proj.bias"]
        # FeedForward は Linear（layers.0）→ GELU（layers.1、重みなし）→ Linear（layers.2）。
        # HFの c_fc が4倍に広げる側、c_proj が元の次元に戻す側
        result[my_prefix + "ff.layers.0.weight"] = state[hf_prefix + "mlp.c_fc.weight"].T
        result[my_prefix + "ff.layers.0.bias"] = state[hf_prefix + "mlp.c_fc.bias"]
        result[my_prefix + "ff.layers.2.weight"] = state[hf_prefix + "mlp.c_proj.weight"].T
        result[my_prefix + "ff.layers.2.bias"] = state[hf_prefix + "mlp.c_proj.bias"]

    return result


def build_vocab(vocab_path: Path, merges_path: Path) -> list[str]:
    # vocab.json のキーはUTF-8のバイトを表示用の文字に置き換えた形（「日」は "æĹ¥"）
    # なので、そのまま使わずにIDごとに復号して元の文字列に戻す。6,000語彙のほとんどは
    # 1文字になる。1文字にならないバイト片のtokenは "�" になって重複するが、
    # 1文字ずつ引く encode で当たることはない
    bpe = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
    hf_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab: list[str] = []
    for token_id in range(len(hf_vocab)):
        vocab.append(bpe.decode([token_id]))

    # CharTokenizer は <|endoftext|>（文書の区切り）と <|unk|>（語彙にない文字）の名前で
    # 特殊tokenを探す。IDは変えず、名前だけHFの [UNK] と </s> から付け替える
    vocab[hf_vocab["[UNK]"]] = UNKNOWN
    vocab[hf_vocab["</s>"]] = END_OF_TEXT
    return vocab


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hugging FaceのGPT-2を自作モデルの形に変換して runs/ に保存する"
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    args = parser.parse_args()

    paths = download_files(args.model_id)
    hf_config = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    config = build_config(hf_config)
    # model.safetensors を「パラメータ名 → テンソル」の辞書として読む
    hf_state = load_file(str(paths["model.safetensors"]))
    state = convert_state_dict(hf_state, config)

    vocab = build_vocab(paths["vocab.json"], paths["merges.txt"])
    assert len(vocab) == state["tok_emb.weight"].shape[0]

    model = JpCharGPT2(len(vocab), config)
    model.load_state_dict(state, strict=True)
    tokenizer = JpCharGPT2Tokenizer(vocab)

    run_dir = RUNS_DIR / args.run_name
    save_run(run_dir, model, config, tokenizer)

    parameter_count = 0
    for parameter in model.parameters():
        parameter_count += parameter.numel()
    size_mb = (run_dir / "model.pt").stat().st_size / 1e6
    print("パラメータ数:", parameter_count)
    print("保存先:", run_dir)
    print(f"model.ptのサイズ: {size_mb:.1f}MB")


if __name__ == "__main__":
    main()
