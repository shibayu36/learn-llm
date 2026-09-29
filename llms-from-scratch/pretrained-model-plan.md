# 学習済みモデルの読み込み計画（ku-nlp/gpt2-small-japanese-char）

## 目的と方針

Hugging Faceで公開されている日本語1文字単位のGPT-2 smallを、自作の部品で組んだモデルに読み込んで動かす。本の5.5「OpenAIの重みを読み込む」を日本語文字モデルで行うことに相当する。

目的は2つ（2026-09-29に決定）。

1. 可視化ツール（Attentionの重み・hidden state・token embedding）を、171GBの日本語で学習の進んだモデルにかけ、Stage 4・5と自作の `l12h12-d768-s5000-lr1e-3` の観察と比べる
2. fine-tuning（付録D・第7章）の起点にする。ゼロから学習した自作モデルではfine-tuning前後の差が読みにくい

決めたこと（理由は `progress.md` の「作業の決め事」）。

- モデルは `GPT` を拡張せず、別クラス `JpCharGPT2` にする
- 重みは変換スクリプトで `runs/` の形式に保存し、以後は既存の `generate`・`evaluate`・`visualize-*` をそのまま使う
- tokenizerは公式の語彙を復号して `CharTokenizer` の語彙にする

作業項目と進捗は `progress.md` の「Stage 6：学習済みモデルの読み込み」。

## 対象モデル

ku-nlp/gpt2-small-japanese-char: https://huggingface.co/ku-nlp/gpt2-small-japanese-char

| 項目 | 値 |
|---|---|
| アーキテクチャ | `GPT2LMHeadModel`（`model_type: gpt2`） |
| 層数 / 次元 / ヘッド数 | 12 / 768 / 12（ヘッドあたり64次元） |
| 位置埋め込み `n_positions` | 1,024 |
| 語彙数 | 6,000（うち5,992が1文字。残りは特殊token4つとバイト片） |
| 活性化関数 | `gelu_new`（tanh近似。自作 `GELU` と同じ） |
| LayerNorm eps | 1e-5（自作 `LayerNorm` と同じ） |
| dropout | 0.1（推論では無効） |
| 学習データ | Wikipedia + CC-100 + OSCARの日本語171GB。A100 1枚で約3か月 |
| ライセンス | モデルカードを参照 |

使うファイル（`https://huggingface.co/ku-nlp/gpt2-small-japanese-char/resolve/main/<file>`）。

| ファイル | 大きさ | 用途 |
|---|---|---|
| `model.safetensors` | 374MB | 重み |
| `config.json` | 763B | 層数・次元・ヘッド数・`n_positions` |
| `vocab.json` | 105KB | 語彙（byte-level BPEの表記） |
| `merges.txt` | 45KB | BPEの結合規則。復号と一致確認に使う |

特殊tokenは `vocab.json` でID 0 `[UNK]`、1 `[PAD]`、2 `<s>`、3 `</s>`。`config.json` の `eos_token_id` は2だが `vocab.json` と食い違うので、文書区切りには `</s>`（ID 3）を使う。

## 自作 `GPT` との差と埋め方

| 項目 | 自作 `GPT` | ku-nlp GPT-2 | 埋め方 |
|---|---|---|---|
| Q/K/Vのbias | なし | あり | `CausalAttention` に `qkv_bias` 引数を足し、`JpCharGPT2` だけTrueにする |
| Q/K/Vの持ち方 | ヘッドごとに別の `nn.Linear` | 1つの `c_attn`（768×2,304）にQ・K・V・全ヘッド分を連結 | 変換時に切り分ける（下記） |
| 出力ヘッド | `out_head` が独立 | `wte` と同じ行列（重み共有） | `JpCharGPT2` で `out_head.weight` に `tok_emb.weight` を代入して共有する |
| 位置埋め込み | `context_length` 行 | 1,024行 | `context_length=1024` で組む |
| Linearの重みの向き | `[out, in]` | Conv1D `[in, out]` | 転置する（本の `load_weights_into_gpt` と同じ） |

Attentionのスケーリング（√64で割る）、Pre-LayerNorm、ショートカット接続の順序、FFNの4倍幅は同じ。

## 実装

### 1. `CausalAttention`・`MultiHeadAttention` に `qkv_bias` 引数

`CausalAttention(d_in, d_out, context_length, qkv_bias=False)` にして `W_query`・`W_key`・`W_value` の `nn.Linear` に渡す。`MultiHeadAttention` も同じ引数を受けて各ヘッドに渡す。既定Falseなので `GPT`・`OneHeadGPT`・`OneLayerOneHeadGPT` の挙動と保存済みrunの読み込みは変わらない。

確認：既存runの `evaluate` で全バッチvalidation lossが `evaluation-results.md` の値と一致すること。

### 2. `JpCharGPT2`

`gpt.py` に `JpCharGPT2(vocab_size, config)` を作る。`GPT` と同じ構造で、違いは `MultiHeadAttention` を `qkv_bias=True` で作ることと、`out_head.weight` に `tok_emb.weight` を代入して重み共有にすることの2点。`LayerNorm`・`FeedForward`・`TransformerBlock`・`MultiHeadAttention` はそのまま使う。

属性名は `GPT` と同じ `tok_emb`・`pos_emb`・`trf_blocks`・`final_norm`・`out_head` にする。`visualize.py` がこの名前と `TransformerBlock`・`MultiHeadAttention` の `isinstance` に依存しているため。

`checkpoint.load_run` に `"JpCharGPT2"` の分岐を足す。

### 3. 変換スクリプト `import_hf_gpt2.py`

`uv run import_hf_gpt2.py` で `runs/ku-nlp-gpt2-small-char/` に `model.pt`・`config.json`・`vocab.json` を書く。引数は `--model-id`（既定 `ku-nlp/gpt2-small-japanese-char`）と `--run-name`（既定 `ku-nlp-gpt2-small-char`）。依存に `safetensors` を足す。

手順。

1. `huggingface_hub.hf_hub_download` で4ファイルを取得する（`~/.cache/huggingface/` に入る）
2. HFの `config.json` から `n_layer`・`n_embd`・`n_head`・`n_positions` を読み、`Config` の `n_layers`・`d_model`・`n_heads`・`context_length` に入れる。学習用の項目（`batch_size`・`steps`・`learning_rate` など）は `Config()` の既定のまま入れる。fine-tuningのときに改めて決める
3. `safetensors.torch.load_file` で重みを読み、下の対応表で `JpCharGPT2` の `state_dict` を組み立てて `load_state_dict(strict=True)` する
4. 語彙を変換して `CharTokenizer` を作る（下記）
5. `save_run` で保存する

重みの対応。HF側のキーは先頭の `transformer.` を外して書く。`h.{i}.attn.bias`・`h.{i}.attn.masked_bias`（Causal maskのbuffer）と `lm_head.weight`（`wte` と同じ）は使わない。`i` は層番号、`h` はヘッド番号、`D=768`、`H=64`。

| HF側 | shape | 自作側 | 処理 |
|---|---|---|---|
| `wte.weight` | [6000, D] | `tok_emb.weight`・`out_head.weight` | 同じテンソルを両方の名前で入れる。重み共有でも `state_dict` には両方の名前が出て、`strict=True` は両方を要求する。`model.pt` には1つ分しか書かれない |
| `wpe.weight` | [1024, D] | `pos_emb.weight` | そのまま |
| `h.{i}.ln_1.weight` / `.bias` | [D] | `trf_blocks.{i}.norm1.scale` / `.shift` | そのまま |
| `h.{i}.attn.c_attn.weight` | [D, 3D] | `trf_blocks.{i}.att.heads.{h}.W_query.weight` など | 列を `[:, :D]`・`[:, D:2D]`・`[:, 2D:]` でQ・K・Vに分け、さらに列を `[:, h*H:(h+1)*H]` でヘッドに分けて転置。[D, H] → [H, D] |
| `h.{i}.attn.c_attn.bias` | [3D] | `...heads.{h}.W_query.bias` など | 同じ切り方で `[h*H:(h+1)*H]` |
| `h.{i}.attn.c_proj.weight` / `.bias` | [D, D] / [D] | `trf_blocks.{i}.att.out_proj.weight` / `.bias` | 転置 / そのまま |
| `h.{i}.ln_2.weight` / `.bias` | [D] | `trf_blocks.{i}.norm2.scale` / `.shift` | そのまま |
| `h.{i}.mlp.c_fc.weight` / `.bias` | [D, 4D] / [4D] | `trf_blocks.{i}.ff.layers.0.weight` / `.bias` | 転置 / そのまま |
| `h.{i}.mlp.c_proj.weight` / `.bias` | [4D, D] / [D] | `trf_blocks.{i}.ff.layers.2.weight` / `.bias` | 転置 / そのまま |
| `ln_f.weight` / `.bias` | [D] | `final_norm.scale` / `.shift` | そのまま |

語彙の変換。

- `tokenizers.ByteLevelBPETokenizer(vocab.json, merges.txt)` を作り、ID 0〜5,999を1つずつ `decode([id])` して文字列に戻す。`vocab.json` のキーはUTF-8のバイトを表示用の文字に置き換えた形（「日」は `æĹ¥`）なので、キーをそのまま使わない
- ID 0 `[UNK]` を `<|unk|>`、ID 3 `</s>` を `<|endoftext|>` に付け替える。`CharTokenizer` がこの名前で特殊tokenを探すため。IDは変えない。ID 1 `[PAD]`・ID 2 `<s>` はそのまま残す（複数文字なのでencodeで当たらない）
- バイト片のtokenは復号すると `�` になる。同じ文字列がIDに重複するが、encodeで当たることはないので気にしない
- `CharTokenizer(vocab)` を作り、`save_run` で `vocab.json`（リスト形式）として保存する

公式tokenizerとの差は、語彙にない文字がPython実装の公式（transformers 4.xの `GPT2Tokenizer`）ではバイト片に分かれ、変換後は `<|unk|>` になる点だけ。半角スペースと改行はバイト片に分けても語彙になく、Python実装の公式でも `[UNK]` になるので、変換後の `<|unk|>` と一致する。モデルカードが全角スペースを使うよう注意しているのはこのため。タブと全角スペースは語彙にあるので、どちらでも残る。

Rust実装の公式（`GPT2TokenizerFast`、transformers 5系の `GPT2Tokenizer`）は、BPEの `unk_token` が設定されず、語彙にないtokenを黙って落とすので、半角スペースと改行が消える。事前学習で使われたのはPython実装と見られる（下の「注意」の `</s> [UNK] <s>` の観察）。一致確認はPython実装相当（`unk_token` を `[UNK]` にした `tokenizers` のBPE）と比べる。

### 4. 一致確認（`tmp/` に置き、commitしない）

- 重み：`uv run --with transformers python tmp/check_jp_char_gpt2.py`。`GPT2LMHeadModel.from_pretrained` で本物を読み、公式tokenizerでencodeした数文（`data/hidden-state-sentences.json` の6文など）を両方に入れ、logitsの最大絶対差が1e-4未満であること。greedyで30文字生成した結果も一致すること
- tokenizer：評価prompt（`data/evaluation-prompts.json`）とhidden state用の6文について、変換後の `CharTokenizer.encode` が公式の `encode(...).ids` と一致すること

### 5. 使い方

```
uv run import_hf_gpt2.py
uv run main.py generate --run-name ku-nlp-gpt2-small-char "日本の首都は"
uv run main.py evaluate --run-name ku-nlp-gpt2-small-char
uv run main.py visualize-attention --run-name ku-nlp-gpt2-small-char
uv run main.py visualize-hidden-states --run-name ku-nlp-gpt2-small-char
uv run main.py visualize-embeddings --run-name ku-nlp-gpt2-small-char
```

注意。

- `evaluate` は文書を `<|endoftext|>`（`</s>`）で区切って窓1,024で測る。事前学習の文書境界は `</s> [UNK] <s>` だったと見られ（次項）、`evaluate` の区切り方とは違うので、validation lossは自作runと同じ条件の参考値として読む。語彙が違う（6,000と4,052）ので自作runのlossとは直接比べない
- `evaluate` を回す前に、このrunだけ半角スペース・改行を除いてencodeするか、そのまま測って参考値と割り切るかを決める。そのまま測ると、改行の直後と文書の先頭文字の位置でlossが大きく出るため。validationテキストには改行10,003個・半角スペース5,647個（全文字の2.4%）があり、評価prompt 09・10にも改行がある。変換後のtokenizerでは、改行と半角スペースはどちらも `<|unk|>` になる。これは事前学習時のtokenizerと同じだが、事前学習では改行は文書の境界にしか現れなかったと見られる。変換後のモデルで確かめると、`<|endoftext|>`（`</s>`）の次は `<|unk|>` を確率1.000で、`<|unk|>` の次は `<s>` を確率1.000で予測する。文中の改行由来の `<|unk|>` の直後も `<s>` が1.000になる。1行1文書のテキストをPython実装でtokenizeすると行間の改行が `[UNK]` になり、`</s> [UNK] <s>` が文書境界の決まった並びになる。つまり `<|unk|>` は「文書の境界」を意味するtokenとして学習されていて、このモデルは文中の改行を知らない。`evaluate` では `</s>` の直後に文書の先頭文字が来るので、そこでも予測が外れる
- モデルカードの使用例はpromptを `<s>` で始めている（`"<s>昨日私は京都で"`）。事前学習の文書はすべて `<s>` から始まるので、`generate`・`evaluate` でpromptや文書の先頭に `<s>`（ID 2）を付けるかを、`evaluate` を回す前に上の改行の扱いと一緒に決める。`CharTokenizer` は `<s>` を1tokenとしてencodeできないので、付けるならencode後のID列の先頭に足す形になる
- モデルカードの eval loss は 1.597（各コーパスから5,000文書ずつ）。`evaluate` の値と比べるときの目安にする。データも区切り方も違うので、桁が合っているかを見る程度
- `visualize-hidden-states` の近傍探索は `config.context_length`=1,024の窓で回る。同じ12層で窓256のときの3分22秒（CPU）より長くかかる見込み。待てなければ窓幅を引数で256にする

## fine-tuningへの接続（第7章を始めるときに決める）

- `train.py` の `train_model` はモデルの種類を問わないので、`JpCharGPT2` をそのまま渡せる。`main.py train` はゼロから作る前提なので、学習済みrunを読んで続きを学習する入口（`finetune` サブコマンドなど）は第7章の設計で決める
- 学習率は事前学習の1e-3より小さくする。本の第7章は5e-5
- 指示データは短いので学習の窓は256のままでよい。1stepは自作 `l12h12` と同じ約0.84秒（B=16・T=256・MPS）
- 事前学習ではdropout 0.1が入っていた。fine-tuningで入れるかは、そのとき決める

## 見直す条件

- mediumやlargeを読みたくなったら `--model-id` を変えるだけ。形はHFの `config.json` から取る。largeの重みは約1.5GB
- rinnaなどSentencePieceのサブワードモデルを読むなら、`CharTokenizer` への変換は成り立たない。`tokenizer.py` に公式tokenizerを包むクラスを足し、`load_run` の戻り値の型を広げる
- Llama系（RoPE・RMSNorm・SwiGLU）を読むなら、部品も対応表も別物になる。このスクリプトはGPT-2アーキテクチャ専用
