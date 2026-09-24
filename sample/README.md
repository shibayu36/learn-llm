# Tiny GPTの参照コード

日本語の文章を使うPart 1の完成コードです。本文は `../docs/part1.md` にあります。このコードを参考にしながら、別の作業ディレクトリに自分で実装していく使い方を想定しています。

## 学習前の生成を試す

`sample/` ディレクトリで実行します。

```bash
uv sync --python 3.14.7 --frozen
uv run --frozen python prepare_data.py
uv run --frozen python preview.py
```

FineWeb-2 Edu Japaneseから1万文書を固定し、trainの9,000文書に出てくる文字を語彙にします。GPTの重みはまだ学習していません。「プログラミングを学ぶには、」の続きを生成し、文字がどう並ぶかを見てください。

保存したデータは次回もそのまま使います。取得条件・文書分割・SHA-256は `dataset-manifest.json` に記録しています。

## 各節の動作確認・実験を試す

`1_2.py`〜`1_12.py` に各節の確認コードがあります。データを用意した後、試したい節のファイルを実行してください。

```bash
uv run --frozen python 1_2.py
uv run --frozen python 1_5.py
```

どのファイルも単独で実行できます。1.11は100回の学習を行い、結果はメモリ上に保持します。1.12は未学習モデルで生成を試します。

`tiny_gpt/` は完成版なので、1.3〜1.5ではその節の計算だけを取り出すため、本文と呼び出し方を変えています。

| ファイル | 完成版での確認方法 |
|---|---|
| `1_3.py` | `model.token_embedding(ids)` でToken Embeddingだけを通す |
| `1_4.py` | tokenと位置のEmbeddingを直接足して比較する |
| `1_5.py` | Q・K・Vを作り、`scaled_attention(..., causal=False)` でMaskなしのAttentionを計算する |

## 学習前後を比較する

```bash
uv run --frozen python 1_13.py
```

標準は1層・1ヘッド、`d_model=128`、context length 128、バッチサイズ16、語彙4,052、約126万パラメータです。10,000回更新し、6つの同じpromptで学習前後のsampling・greedyを比べます。

設定は `tiny_gpt/config.py` にあります。短い動作確認では `steps` を20、`run_name` を `"part1-smoke"` にします。各実行は同じseedで新しいモデルを初期化します。

結果は `runs/<run_name>/` に保存します。同じ名前で実行すると結果を上書きするため、比較するときは名前を変えてください。

| ファイル | 内容 |
|---|---|
| `metrics.json` | 実験条件、パラメータ数、loss、時間、学習前後の生成 |
| `loss.png` | train/validation lossの推移 |
| `model.pt` | 学習済みパラメータとモデル設定 |
| `tokenizer.json` | そのモデルで使った語彙（token IDの順に並べた文字の一覧） |

モデルとTokenizerは組にして使います。`model.pt` は生成・追加学習の出発点となるパラメータです。optimizerの状態を含む完全な学習再開用checkpointではありません。

実測したlossと生成例は `results/README.md` にあります。

## 計算とデータを確認する

先にデータ取得コマンドを実行してください。

```bash
uv run --frozen python -m unittest discover -s tests -v
```

## データの出典

配布元・作成者：Yuichi Tateno / FineWeb2 Edu Japanese: https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese

配布元のライセンスはODC-By 1.0です。元のWeb文書のURLは、保存した各文書の `url` に含めています。
