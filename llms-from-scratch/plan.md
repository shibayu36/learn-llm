# llms-from-scratch の学習計画

## 目的

書籍『つくりながら学ぶ！LLM自作入門』（Sebastian Raschka, Build a Large Language Model (From Scratch) の日本語訳。以下「本」）のコードを参考にしながら、GPTを自分の手で組み立て、部品を1つずつ拡張しながら出力と評価値の変化を観察する。

本のとおりに進めると理解が追いつかなかったので、次の3点を本から変える。

- パラメータ数を減らす。1回の学習をCPUで数分以内に終え、条件を変えた比較を何度も回せるようにする
- 日本語を学習させる。生成結果の変化を自分の目で読み取れるようにする
- Attentionまわりを「1層・1ヘッド → 複数層 → Multi-head」の順に拡張し、段階ごとに出力を見て評価する。本は第3章でMulti-headまで作ってから第4章で層を積むが、ここでは最小構成で学習・生成まで通してから拡張する

`docs/` の自作ハンズオン資料と `tiny-gpt-handson/` は中断した。`docs/handson-plan.md` の制作基準はここでは使わない。ただし、日本語データの固定条件・文字Tokenizer・評価のそろえ方はそこで決めたものを流用する。

## 本との対応

| 本 | ここでの扱い |
|---|---|
| 第2章 テキストデータの準備（2.2〜2.4 Tokenizer、2.6 スライディングウィンドウ、2.7〜2.8 埋め込み） | Stage 1。Tokenizerは本の `SimpleTokenizerV2` の考え方を文字単位に置き換える。2.5 BPE（tiktoken）は使わない。2.6の `GPTDatasetV1` + `DataLoader` はそのまま参考にする |
| 第3章 Attentionメカニズムのコーディング（3.4 学習可能な重み、3.5 Causal Attention） | Stage 0で実装済み（`SelfAttention_v1`、`CausalAttention`） |
| 第3章 3.6 Multi-head Attention | Stage 5。本と同じく `MultiHeadAttentionWrapper`（ヘッドを並べて連結）→ `MultiHeadAttention`（1つの行列を分割）の2段階で作る |
| 第4章 GPTモデルを一から実装する（4.2 LayerNorm、4.3 GELU/FFN、4.4 ショートカット、4.5 Transformerブロック） | Stage 0で実装済み |
| 第4章 4.6 GPTモデル | Stage 0で1層版 `OneLayerOneHeadGPT` を実装済み。層を積む `GPTModel` はStage 4 |
| 第4章 4.7 テキストを生成する | Stage 2。本の `generate_text_simple`（greedy）は作らず、最初から `generate`（`temperature=0` でgreedy）にする |
| 第5章 ラベルなしデータでの事前学習（5.1 評価、5.2 訓練、5.3 デコーディング戦略、5.4 保存と読み込み） | Stage 2〜3。損失関数・訓練ループ・temperature・保存。5.3.2 top-kは学習後の生成を見て必要なら足す |
| 第5章 5.5 OpenAIの重みを読み込む | やらない。語彙もサイズも違うので読み込めない |
| 第6章 分類のためのファインチューニング | やらない（必要になったら再検討） |
| 第7章 指示に従うためのファインチューニング | Stage 6の候補。Multi-headまで終えてから判断する |
| 付録D 訓練ループの高度なテクニック（warmup、cosine減衰、勾配クリッピング） | Stage 6の候補 |
| 付録E LoRA | やらない（必要になったら再検討） |

## 本から変える決定事項

- Tokenizerは1文字1token。trainに現れた文字にEOS・UNKを加えた語彙（約4,052）。tiktokenのGPT-2 BPEは日本語がbyte断片になり生成結果が読めないため使わない
- 学習データは `hotchpotch/fineweb-2-edu-japanese` の `small_tokens_cleaned` から固定した1万文書（train 9,000・validation 1,000）。取得条件は `data/fineweb-japanese-10k/manifest.json`
- 本の `GPTModel` は `n_layers`・`n_heads` を最初から持つが、ここでは1層1ヘッドと分かる `OneLayerOneHeadGPT` を先に作り、層を積む版・Multi-head版は後で別クラスとして足す。`OneLayerOneHeadGPT` は比較の基準として残す
- MLP中間次元は本どおり `4 * d_model` 固定。dropoutなし。Q/K/Vと出力層はbiasなし
- Causal maskは本の `CausalAttention` と同じ `triu(diagonal=1)` + `masked_fill(-inf)`
- 本では `GPT_CONFIG_124M` の辞書だが、ここでは `Config` クラス（型付き）にする
- 訓練ループは本の `train_model_simple` のepoch単位ではなく、`steps` 回の更新で回す。層数・ヘッド数の比較で「同じ回数だけ更新した」とそろえるため。trainは1,447バッチなので5,000stepは約3.5周
- 学習中の評価は、本と同じく訓練セット・検証セットの先頭 `eval_batches` バッチで行う。ただし訓練用の `DataLoader` は `shuffle=True` で毎回違う窓が出るので、評価用に `shuffle=False` の `DataLoader` を別に作り、毎回同じ窓で測る
- optimizerは本と同じ `AdamW`、`weight_decay=0.1` も同じ。学習率は本の3e-4ではなく1e-3。このサイズでは3e-4だと2,000stepでも下がりきらず、1e-3にしても序盤に跳ねなかったため。保存はモデルの `state_dict` だけで、optimizerの状態は保存しない（学習の再開はしない）
- `runs/<run_name>/config.json` には `Config` の値に加えてモデルのクラス名を入れる。Stage 4で `GPTModel` が増えても `generate` がどのクラスを組み立てるか迷わないようにするため
- 実行はCPU固定。device選択のコードは書かない。Mac GPU（MPS）は今のサイズでは効きにくいので、Stage 6でモデルを大きくするときに検討する（変える場所は `main.py` の `.to(device)`、`gpt.py` の `torch.arange(..., device=)`、`calc_loss_batch`、`torch.load(map_location=)` の4か所）
- コードのコメントは本の用語（「Causal Attention」「ショートカット接続」「層正規化」など）に合わせる

## 実行条件

以下はStage 3で記録した64次元モデルの比較基準。追加実験では `config.py` の `d_model` だけを128に変えた（batch 16のまま）。全バッチvalidation lossは3.717512から3.376540へ下がったが、greedyの反復は強まった。結果は `evaluation-results.md` に記録した。現在の設定は128次元で、Stage 4・5で採用する幅は開始前に決める。

| 条件 | 基準値 |
|---|---:|
| `d_model` | 64 |
| `context_length` | 256 |
| `batch_size` | 16 |
| MLP中間次元 | 256（`4 * d_model`） |
| 更新回数 `steps` | 5,000 |
| 学習率 | 1e-3（AdamW） |
| 語彙数 | 4,052 |
| 1層1ヘッドのパラメータ数 | 580,800（うちEmbedding+出力層が535,040） |

パラメータ数の9割はEmbeddingと出力層で、層やヘッドを増やしても1層あたり約46,000しか増えない。層数・ヘッド数の比較で「パラメータ数がほぼ変わらないのに結果が変わるか」を見られる。

`context_length` は当初64にしていたが、日本語で2〜3文しか入らず「話題を保てるか」を見るには短いので、本の第5章の訓練と同じ256にした。1stepの所要時間は64で8ms・256で21ms（1層1ヘッド、CPU）と差が小さい。

更新回数と学習率は当初2,000step・3e-4だったが、lossが下がりきらないまま終わったので、Stage 3で5,000step・1e-3に変えた。1層1ヘッドで約105秒。層を増やすと1stepの時間も増えるので、4層で5分を超えるようなら見直す。

## 評価のそろえ方

評価計画は `evaluation-plan.md` にまとめる。

## ファイル構成

```
llms-from-scratch/
  config.py        Config（Stage 4で n_layers、Stage 5で n_heads を追加）
  gpt.py           モデルの部品とGPT本体（本のgpt.pyに相当）
  tokenizer.py     文字Tokenizer（Stage 1）
  dataset.py       GPTDataset + DataLoader（Stage 1）
  generate.py      generate（temperature付き。Stage 2）
  train.py         損失・評価・訓練ループ（Stage 2〜3）
  evaluate.py      全バッチのloss計算・評価用promptの生成
  plot.py          学習中のloss曲線の描画
  checkpoint.py    学習結果の保存と読み込み（Stage 3）
  main.py          入口。`train`・`generate`・`evaluate` のサブコマンドを持つ
  experiments/     層数・ヘッド数の比較スクリプト（Stage 4〜）
  runs/            実行結果（model.pt、config.json、vocab.json、metrics.json、loss.png）。commitしない。残したい結果は evaluation-results.md に書く
  evaluation-results.md  評価結果。runごとの条件・loss・生成・観察
  docs/            用語集など学習用の資料
  data/            固定した日本語データと評価prompt（evaluation-prompts.json）
  prepare_data.py  データ取得
```

本は章ごとにNotebookで進めるが、ここでは部品をモジュールに分け、`main.py` 1本を実行入口にする。各節の動作確認は `main.py` に書き、次の段階へ進むときに不要な確認コードは消す。

Stage 3以降の `main.py` は次の3つのサブコマンドを持つ。

- `uv run main.py train`：学習して `runs/<run_name>/` にモデル・Config・語彙・評価値を保存する
- `uv run main.py generate "日本の首都は"`：保存済みのモデルと語彙を読み込み、1つのpromptの続きを生成する。学習をやり直さずに何度でも試せる。オプションでtemperatureを指定する
- `uv run main.py evaluate --run-name l1h1`：保存済みモデルの全バッチ評価と固定10promptの生成を行い、数値と生成結果を保存する。比較方法は `evaluation-plan.md` を参照する

`generate` のpromptは1回の実行につき1つにする。固定した複数promptでの評価には `evaluate` を使う。

## 段階

### Stage 0：部品の実装（完了）

本の第3章・第4章に沿って `gpt.py` に `SelfAttention_v1`・`CausalAttention`・`LayerNorm`・`GELU`・`FeedForward`・`TransformerBlock`・`OneLayerOneHeadGPT` を実装した。日本語データも取得済み。

### Stage 1：Tokenizerとデータ（本 2.2〜2.8）

- 文字Tokenizer（`encode`・`decode`、EOS・UNK）
- 全文書をEOSで区切って1本のtoken列にし、本の `GPTDatasetV1` と `create_dataloader_v1` のように窓を切り出す
- 動作確認：短い日本語をencode→decodeして戻ること。1バッチの入力と正解が1文字ずれていること。`OneLayerOneHeadGPT` に通して `[B, T, vocab_size]` が返ること

### Stage 2：生成と損失（本 4.7、5.1）

- `generate`（`temperature=0` でgreedy、`>0` でsampling）で、学習前のモデルから「日本の首都は」の続きを出す。でたらめな文字列になること、greedyは毎回同じでsamplingは毎回違うことを確認する
- 損失（cross entropy）を1バッチで計算し、学習前の値が `ln(vocab_size)` 付近であることを確認する

### Stage 3：1層1ヘッドの学習（本 5.2〜5.4）

- 訓練ループ（AdamW）、一定stepごとのtrain/validation loss、学習中の生成サンプル表示
- 学習後のモデル・Config・語彙の保存と読み込み。語彙はモデルと組で保存し、読み込み時はtrainを読み直さず保存した語彙を使う
- `main.py` を `train` / `generate` / `evaluate` のサブコマンドに分ける
- 基準の記録：パラメータ数、loss曲線、学習前後の生成、所要時間
- 観察：学習前のでたらめな文字列から、日本語の語句や文末のつながりが現れるか。話題を保った文章になっているか

### Stage 4：複数層（本 4.6）

- Configに `n_layers` を追加し、`TransformerBlock` を `nn.Sequential` で積む `GPTModel` を作る（本の `GPTModel` と同じ形。Attentionは1ヘッドのまま）
- `n_layers=1` の `GPTModel` が `OneLayerOneHeadGPT` と同じパラメータ数・同じ動きになることを確認する
- 実験：1・2・4層を同じ条件で学習し、パラメータ数・loss・生成・時間を並べる
- 観察：層を増やせばlossが下がるか。下がらないなら容量不足かデータ不足か学習不足か。生成の質は変わるか
- 任意：ショートカット接続を外すと深い層で学習が進まなくなるか（本 4.4の実験を学習で再現）

### Stage 5：Multi-head Attention（本 3.6）

- まず `MultiHeadAttentionWrapper`（`CausalAttention` を `n_heads` 個並べて連結）を作り、出力の次元が `n_heads * d_out` になることを確認する
- 次に `MultiHeadAttention`（1つの `W_query` などを `n_heads` に分割し、`view`・`transpose` で並列計算し、`out_proj` で戻す）を作り、`d_model` を固定したままヘッド数だけ変えられるようにする
- Configに `n_heads` を追加し、`GPTModel` のAttentionを `MultiHeadAttention` に置き換える。`n_heads=1` で結果が変わらないことを確認する
- 実験：Stage 4で採用した `d_model` に固定して1・2・4ヘッドを比べる。Q/K/Vのパラメータ数がヘッド数で変わらないことを確認する
- 観察：ヘッドごとのAttention weightを同じ入力で並べ、ヘッドによって見る位置が違うか。lossと生成は変わるか

### Stage 6：その後の候補

Stage 5まで終えてから、次のどれをやるか決める。

- 付録D：warmup・cosine減衰・勾配クリッピングを入れて学習が安定・改善するか
- 第7章：小さな日本語の指示データでInstruction Tuning
- `d_model`・`context_length`・更新回数を増やしたときの変化

## この計画を見直す条件

- 1回の学習が5分を超えるようになったら、更新回数か `d_model` を下げる
- 語彙やデータを変えるなら、Tokenizerとモデルを組にして作り直す
- 実験用の分岐が `gpt.py` に増えて読みにくくなったら、基準のモデルと実験スクリプトの責務を分ける
