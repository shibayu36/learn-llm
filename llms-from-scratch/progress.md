# llms-from-scratch の進捗

計画は `plan.md`。ここでは各Stageの作業状態と、実行して分かったことを記録する。チェックは実行して確認できたときに付ける。

## 現在の作業

Stage 1（Tokenizerとデータ）から始める。Stage 0の部品は実装済みだが、まだデータを通して動かしていない。

## Stage 0：部品の実装

- [x] 日本語データ1万文書を取得し、train/validationに分けて固定する（`prepare_data.py`、2026-09-25）
- [x] `Config`（d_model 64、context_length 64、batch 16、steps 2000）
- [x] `SelfAttention_v1`（本 3.4）
- [x] `CausalAttention`（本 3.5）
- [x] `LayerNorm`・`GELU`・`FeedForward`・`TransformerBlock`（本 4.2〜4.5）
- [x] `OneLayerOneHeadGPT`（本 4.6の1層版）
- [x] `pyproject.toml` の `name` を `tiny-gpt-handson` から `llms-from-scratch` に直す（`uv lock` も更新、2026-09-25）
- [ ] `config.py`・`gpt.py` をcommitする

## Stage 1：Tokenizerとデータ

- [ ] `tokenizer.py`：文字Tokenizer（EOS・UNK）
- [ ] `dataset.py`：EOSでつないだtoken列から窓を切り出すDataset・DataLoader
- [ ] `main.py`：encode→decodeの往復、1バッチの入力と正解のずれ、モデル出力のshapeを確認
- [ ] 語彙数・train/validationのtoken数を記録する

## Stage 2：生成と損失

- [ ] `generate.py`：`generate_text_simple`（greedy）
- [ ] 学習前のモデルで「日本の首都は」の続きを生成し、でたらめな文字列を確認
- [ ] `train.py`：1バッチの損失を計算し、学習前の値が `ln(vocab_size)` 付近であることを確認

## Stage 3：1層1ヘッドの学習

- [ ] 訓練ループ・定期評価・学習中の生成サンプル
- [ ] temperature・top-kのsampling
- [ ] モデル・Config・語彙の保存と読み込み
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

（各Stageの実測値・観察をここに追記する）
