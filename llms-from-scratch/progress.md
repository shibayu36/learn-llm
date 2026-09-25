# llms-from-scratch の進捗

計画は `plan.md`。ここでは各Stageの作業状態と、実行して分かったことを記録する。チェックは実行して確認できたときに付ける。

## 現在の作業

Stage 1（Tokenizerとデータ）は完了。次はStage 2の `generate.py`（greedyの `generate_text_simple`）と、1バッチの損失計算から始める。

- Stage 1までのコードと記録はすべてcommit済み
- `main.py` は現在Stage 1の動作確認コード。Stage 2に進むときは、Tokenizerと1バッチの確認は残し、Stage 2の確認を末尾に足す。Stage 3で `train` / `generate` のサブコマンド構成に変える（`plan.md` 参照）

## Stage 0：部品の実装

- [x] 日本語データ1万文書を取得し、train/validationに分けて固定する（`prepare_data.py`、2026-09-25）
- [x] `Config`（d_model 64、context_length 256、batch 16、steps 2000。context_lengthは64から256に変更、2026-09-25）
- [x] `SelfAttention_v1`（本 3.4）
- [x] `CausalAttention`（本 3.5）
- [x] `LayerNorm`・`GELU`・`FeedForward`・`TransformerBlock`（本 4.2〜4.5）
- [x] `OneLayerOneHeadGPT`（本 4.6の1層版）
- [x] `pyproject.toml` の `name` を `tiny-gpt-handson` から `llms-from-scratch` に直す（`uv lock` も更新、2026-09-25）
- [x] `config.py`・`gpt.py` をcommitする（c071c18）

## Stage 1：Tokenizerとデータ

- [x] `tokenizer.py`：文字Tokenizer（EOS・UNK、JSONで保存と読み込み、2026-09-25）
- [x] `dataset.py`：EOSでつないだtoken列から窓を切り出すDataset・DataLoader（2026-09-25）
- [x] `main.py`：encode→decodeの往復、未知の文字が `<|unk|>` になること（2026-09-25）
- [x] `main.py`：1バッチの入力と正解のずれ、モデル出力のshapeを確認（2026-09-25）
- [x] 語彙数・train/validationのtoken数を記録する（2026-09-25）

## Stage 2：生成と損失

- [ ] `generate.py`：`generate_text_simple`（greedy）
- [ ] 学習前のモデルで「日本の首都は」の続きを生成し、でたらめな文字列を確認
- [ ] `train.py`：1バッチの損失を計算し、学習前の値が `ln(vocab_size)` 付近であることを確認

## Stage 3：1層1ヘッドの学習

- [ ] 訓練ループ・定期評価・学習中の生成サンプル
- [ ] temperature・top-kのsampling
- [ ] モデル・Config・語彙の保存と読み込み（`runs/l1h1/` に model.pt・config.json・vocab.json）
- [ ] `main.py` の `train` / `generate` サブコマンド
- [ ] 基準の記録（`runs/l1h1/metrics.json`）：パラメータ数・loss曲線・学習前後の生成・所要時間
- [ ] 所要時間を見て `steps` を見直す

## Stage 4：複数層

- [ ] `Config.n_layers` と `GPTModel`
- [ ] `n_layers=1` が `OneLayerOneHeadGPT` と一致することを確認
- [ ] 1・2・4層の比較（`experiments/`）
- [ ] 結果と考察を記録

## Stage 5：Multi-head Attention

- [ ] `MultiHeadAttentionWrapper`
- [ ] `MultiHeadAttention`（分割方式）と `Config.n_heads`
- [ ] `n_heads=1` で結果が変わらないことを確認
- [ ] 1・2・4ヘッドの比較（`experiments/`）
- [ ] ヘッドごとのAttention weightの観察
- [ ] 結果と考察を記録

## Stage 6：その後の候補

- [ ] Stage 5の結果を見て、付録D・第7章・モデル拡大のどれをやるか決める

## 実行して分かったこと

### Stage 1：Tokenizer（2026-09-25）

- 語彙数 4,052（trainに現れた文字 4,050 + `<|endoftext|>` + `<|unk|>`）。計画どおり
- token数：train 5,920,916、validation 656,568
- `<|unk|>` の数：train 0、validation 96（validationにしか出ない文字が96文字分ある）
- 1文字1tokenなので、1stepで見る文字数は `batch 16 × context 256 = 4,096`。2,000stepでtrain全体を約1.4周する計算

### Stage 1：データ（2026-09-25）

- `<|endoftext|>` を挟んだ後のtoken数：train 5,929,916（文書数9,000ぶん増えた）、validation 657,568
- `context_length=256`・`stride=256` の窓：train 23,163個（batch 16で1,447バッチ）、validation 2,568個（161バッチ）
- 1バッチは `inputs [16, 256]`・`targets [16, 256]`。inputsの1文字後ろがtargetsになっていることを確認
- `OneLayerOneHeadGPT` の出力は `[16, 256, 4052]`、パラメータ数 580,800（位置埋め込みが 256×64 になり、context 64のときの568,512から12,288増えた）
- context_lengthの検討：1stepの時間は 64で8ms・128で12ms・256で21ms。計算時間は制約にならないので、話題の持続を見やすい256にした
