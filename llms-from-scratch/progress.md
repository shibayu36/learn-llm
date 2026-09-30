# llms-from-scratch の進捗

計画は `plan.md`。ここでは各Stageの作業状態と、実行して分かったことを記録する。チェックは実行して確認できたときに付ける。

## 現在の作業

Stage 7「KVキャッシュ」を進めている（2026-09-29に開始。計画は `kv-cache-plan.md`）。段階4の `generate --kv-cache` まで完了。次は段階5の `kv-cache-results.md` の残り、ku-nlpのキャッシュの大きさと3つの問いへの考察（生成時間の比較は記録済み）。

Stage 6「学習済みモデルの読み込み」の作業項目はすべて完了（2026-09-29）。ku-nlpの日本語1文字GPT-2 smallを `JpCharGPT2` に読み込み、評価を `evaluation-results.md` の「学習済みモデル」、可視化3種の観察を `visualization-results.md` の「学習済みモデル」に記録した。

Stage 5（Multi-head Attention）は、4層・128次元で1ヘッド（`l4h1-d128-s5000-lr1e-3-outproj`）と4ヘッド（`l4h4-d128-s5000-lr1e-3`）を学習・評価し、`evaluation-results.md` の「ヘッド数」に記録した。4ヘッドのrunでのヘッドごとのAttention weightの観察は未着手。Stage 6の下見としてGPT-2 smallの形（`l12h12-d768-s5000-lr1e-3`）も学習し、`evaluation-results.md` の「モデル規模」に記録した。同runの `visualize-hidden-states`・`visualize-embeddings` は生成済みで、近傍コーパスをvalidation全体に広げた（下記）。観察は `visualization-results.md` には記録していない（「学習済みモデル」の節は学習済みモデルだけを読む）。`config.py` は `d_model=128`・`n_layers=4`・`n_heads=4`・`steps=5000` に戻してある。固定10promptの学習前出力は未記録。

## 作業の決め事

- `evaluation-results.md` は「変えた条件のカテゴリ（層数・幅・更新回数と学習率）を `##`、問いへの答えが読める見出しを `###`」の構成にする。各節は結論の段落、設定の段落、詳細の順に書く。新しいカテゴリ・節は上に足す。各runの学習曲線と全20出力は末尾にまとめず、そのrunを最初に使った問いの節の末尾に置く。計画外の参考実験は1段落に収める
- `visualization-results.md` も同じ構成にする。変えた条件（層数・幅）を `##`、「変えたら内部表現がどう変わったか」を `###` にし、1つの節でhidden stateとtoken embeddingの両方を使う。条件を変える前の1モデルだけの観察は「基準モデル」のカテゴリに置く。見出しに日付は書かない（別の日にやっても条件は変わらない）。節の見出しは結論が読めるように書き、詳細な表は見出しと最初の段落のあとに置く
- `runs/` をcommitしないのは、model.pt を入れないなら config.json や metrics.json だけ残しても再現できないため
- `load_run` は復元だけを行い、`model.eval()` は生成・評価する側で呼ぶ。読み込んだモデルを何に使うかは呼び出し側が決めることなので
- 保存・読み込みは `train.py` ではなく `checkpoint.py` に分けた。`generate` が `train.py` を読み込むと「生成に訓練が要る」ように見えるため
- 新しい用語が出る節では用語集 `docs/glossary.md` に追記する
- hidden stateの「近い文脈」はvalidation全体から探し、「同じ2文字」「同じ文字・前が違う」「別の文字」の3列に分けて出す。先頭4万位置では対象の熟語が1〜2回しか出ず、全体にすると同じ単語だけで上位10が埋まるため
- hidden stateの「偶然の水準」は「別の文字」の位置だけで取る。全位置で取ると「の」のように数万回出る文字では上位0.1%が同じ文字で埋まり、水準にならないため
- optimizerは本と同じく `train_model` の外で作って渡す形のままにする。モデルとoptimizerは `.grad` を通して暗黙につながるので、optimizerは必ずそのモデルの `parameters()` で作る。Stage 6でスケジューラやパラメータのグループ分けを入れるときも外で作る
- 学習済みモデルは `GPT` を拡張せず、別クラス `JpCharGPT2` にする。`GPT` は教材の本体でStageごとに変わりうるので、固定して使う学習済みモデルと切り離す。`LayerNorm`・`FeedForward`・`TransformerBlock` は共有し、違いはQ/K/Vのbiasと出力ヘッドの重み共有だけにする
- Hugging Faceの重みは変換スクリプトで `runs/` の形式（model.pt・config.json・vocab.json）に保存し、`load_run` には変換の処理を入れない。既存の `generate`・`evaluate`・`visualize-*` がそのまま使えるため
- Causal maskのbufferは `persistent=False` にして `model.pt` に保存しない。`context_length` から作り直せる固定の表で、ヘッドごとに1つ持つので12層12ヘッド・窓1,024では144個・604MBになるため
- ku-nlpのtokenizerは公式の `vocab.json` を復号して `CharTokenizer` の語彙にする。6,000語彙のうち5,992が1文字で、実質的に文字表のため。`</s>`（ID 3）を `<|endoftext|>`、`[UNK]`（ID 0）を `<|unk|>` に付け替え、IDは変えない。公式（transformers 4.xのPython実装 `GPT2Tokenizer`）との差は、語彙にない文字が公式ではバイト片に分かれ、変換後は `<|unk|>` になる点だけ。半角スペースと改行は公式でも `[UNK]` になり、変換後と一致する。Rust実装の公式（`GPT2TokenizerFast`、transformers 5系）は語彙にないtokenを黙って落とすので、半角スペースと改行が消える。promptでは語彙にある全角スペースを使う。改行と半角スペースの扱いは次項
- ku-nlpのrunでは、改行と半角スペースを `encode` の前に取り除く `JpCharGPT2Tokenizer` を使い、`load_run` が `config.json` の `"model"` で選ぶ。このモデルは `<|unk|>` を文書の境界として覚えていて、文中の改行を `<|unk|>` にするとvalidation lossの4割がその周辺から出るため
- ku-nlpのrunでは、promptや文書の先頭に `<s>` を付けない。lossの改善が0.01程度で、付けると可視化の対象位置がずれるため

## Stage 0：部品の実装

- [x] 日本語データ1万文書を取得し、train/validationに分けて固定する（`prepare_data.py`、2026-09-25）
- [x] `Config`（context_lengthは64から256に変更、2026-09-25）
- [x] `SelfAttention_v1`（本 3.4）
- [x] `CausalAttention`（本 3.5）
- [x] `LayerNorm`・`GELU`・`FeedForward`・`TransformerBlock`（本 4.2〜4.5）
- [x] `OneLayerOneHeadGPT`（本 4.6の1層版）

## Stage 1：Tokenizerとデータ

- [x] `tokenizer.py`：文字Tokenizer（EOS・UNK、JSONで保存と読み込み、2026-09-25）
- [x] `dataset.py`：EOSでつないだtoken列から窓を切り出すDataset・DataLoader（2026-09-25）
- [x] `main.py`：encode→decodeの往復、未知の文字が `<|unk|>` になること（2026-09-25）
- [x] `main.py`：1バッチの入力と正解のずれ、モデル出力のshapeを確認（2026-09-25）
- [x] 語彙数・train/validationのtoken数を記録する（2026-09-25）

## Stage 2：生成と損失

- [x] `generate.py`：`generate`（`temperature=0` でgreedy、`>0` でsampling）と `text_to_token_ids`・`token_ids_to_text`（2026-09-25）
- [x] 学習前のモデルで「日本の首都は」の続きを生成し、でたらめな文字列を確認。T=0は毎回同じ、T=0.8は毎回違うことも確認（2026-09-25）
- [x] `train.py`：1バッチの損失を計算し、学習前の値が `ln(vocab_size)` 付近であることを確認（2026-09-25）

## Stage 3：1層1ヘッドの学習

- [x] 訓練ループ・定期評価・学習中の生成サンプル（`train.py` の `calc_loss_loader`・`evaluate_model`・`generate_and_print_sample`・`train_model`、2026-09-25）
- [x] モデル・Config・語彙の保存と読み込み（`checkpoint.py` の `save_run`・`load_run`。`runs/l1h1/` に model.pt・config.json・vocab.json。config.jsonにはモデルのクラス名 `OneLayerOneHeadGPT` も入る、2026-09-26）
- [x] `main.py` の `train` / `generate` サブコマンドと `--run-name`（2026-09-26）
- [x] 基準の記録（各runの `metrics.json`）：パラメータ数・学習中のloss履歴・学習後の20出力（10promptのgreedyとID04のsampling 10回）・所要時間（2026-09-26）
- [x] `main.py evaluate`：全バッチのvalidation loss、固定10promptの生成、結果保存（2026-09-26）
- [x] 学習曲線 `loss.png` と `evaluation-results.md` の最終loss比較表（2026-09-26）
- [x] 既存2runを保存済みConfigで再学習・評価し、結果を `evaluation-results.md` に記録（2026-09-26）
- [x] 所要時間を見て `steps` を見直す（5,000step・学習率1e-3を新しい基準にした、2026-09-26）
- [x] `d_model` だけ64から128に増やし、全バッチloss・固定20出力・学習時間を比較する。batch sizeは16のまま（2026-09-26）
- [x] Token embeddingの地図（`visualize.py`・`main.py visualize-embeddings`・`templates/embedding_map.html`）。結果は `visualization-results.md`（2026-09-26）
- [x] Hidden stateの変化（`visualize.py` の `build_hidden_state_tables`・`main.py visualize-hidden-states`・`templates/hidden_states.html`）。結果は `visualization-results.md`（2026-09-26）

## Stage 4：複数層

- [x] `Config.n_layers` と `OneHeadGPT`。`main.py train` と `checkpoint.py` を対応させる（2026-09-27）
- [x] `n_layers=1` が `OneLayerOneHeadGPT` と一致することを確認（同seedのパラメータ・logits・勾配、既存runの重みを流し込んだ全バッチloss。`tmp/check_gpt_model.py`、2026-09-27）
- [x] 4層を学習・評価し、1層の基準runと比較（`l4h1-d128-s5000-lr1e-3`、`evaluation-results.md`、2026-09-27）
- [x] 1層を20,000stepで学習し、4層5,000stepと比べる（`l1h1-d128-s20000-lr1e-3`、2026-09-27）
- [x] 4層のrunで層ごとのhidden stateを観察し、128次元のembeddingも1層と比較（`visualization-results.md`、2026-09-27）
- [x] 結果と考察を記録（`evaluation-results.md`・`visualization-results.md`、2026-09-27）

2層の学習は行わない。1層と4層の差で層を積む効果は読めたので、Stage 5でも4層だけで比べる（2026-09-27）。

## Stage 5：Multi-head Attention

- [x] `MultiHeadAttention`（`CausalAttention` を並べて連結 + `out_proj`）と `Config.n_heads`。shape・パラメータ数・1ヘッドが `CausalAttention` と一致することを `tmp/check_multihead.py` で確認（2026-09-27）
- [x] `TransformerBlock` にAttentionを外から渡す形に変える。既存6runの `config.json` に `"n_heads": 1` を足し、旧コードと同seedの初期値・logits、保存済みrunの全バッチloss（3.376540）と生成が一致することを確認（2026-09-27）
- [x] `GPT`（`OneHeadGPT` のAttentionを `MultiHeadAttention` にしたもの）を作り、`checkpoint.load_run`・`main.py train` を対応させる（2026-09-28）
- [x] `n_heads=1` の `GPT` に `OneHeadGPT` の重みを流し込み、`out_proj` を単位行列にすると一致することを確認。同seed初期値のlogitsは最大差0.0、4層runの全バッチvalidation lossは保存値 3.067810 と一致。パラメータ数の差は `out_proj` 4層分の 66,048。`tmp/check_gpt_model_heads.py`（2026-09-28）
- [x] 4層で1ヘッドと4ヘッドを比較（`l4h1-d128-s5000-lr1e-3-outproj`・`l4h4-d128-s5000-lr1e-3`。`evaluation-results.md` の「ヘッド数」、2026-09-28）
- [ ] ヘッドごとのAttention weightの観察
- [ ] 結果と考察を記録

## Stage 6：その後の候補

- [x] モデル拡大の下見：GPT-2 smallの形（12層・12ヘッド・768次元）を5,000step学習し、lossと生成と過剰適合の出方を見た（`l12h12-d768-s5000-lr1e-3`、2026-09-28）
- [ ] Stage 5の結果を見て、付録D・第7章・モデル拡大のどれをやるか決める

## Stage 6：学習済みモデルの読み込み（ku-nlp/gpt2-small-japanese-char）

171GBの日本語で学習された1文字単位のGPT-2 small（12層・12ヘッド・768次元）を自作コードに読み込む。目的は2つ（2026-09-29に決定）。

- 可視化ツールを学習の進んだモデルにかけ、Stage 4・5と自作の `l12h12-d768-s5000-lr1e-3` の観察と比べる
- fine-tuning（付録D・第7章）の起点にする。ゼロから学習した自作モデルではfine-tuning前後の差が読みにくい

手元で回せる根拠は、同じ形の `l12h12-d768-s5000-lr1e-3` の実績。学習は1step 0.84秒（B=16・T=256・MPS）なので第7章の規模（約1,100件を2周、300step弱）なら数分、hidden stateの近傍探索は3分22秒（CPU）。本の5.5「OpenAIの重みを読み込む」を日本語文字モデルで行うことに相当する。作り方（重みの対応表・語彙の変換・一致確認の方法）は `pretrained-model-plan.md`。

- [x] `CausalAttention`・`MultiHeadAttention` に `qkv_bias` 引数（既定False）を足す。`l4h4-d128-s5000-lr1e-3`（`GPT`）と `l4h1-d128-s5000-lr1e-3`（`OneHeadGPT`）を `evaluate` し直し、全バッチvalidation loss（3.094512・3.067810）と全20出力が変更前の `metrics.json` と一致した（2026-09-29）
- [x] `gpt.py` に `JpCharGPT2(vocab_size, config)` を作る。`GPT` と同じ構造でQ/K/Vにbiasあり、出力ヘッドはトークン埋め込みと重み共有。`checkpoint.load_run` に分岐を足す。GPT-2 smallの形（語彙6,000）でパラメータ数が `GPT` より bias分 27,648 多く共有分 4,608,000 少ない 90,450,432 になること、logitsが `final_norm` 後のベクトルと埋め込み表の内積に一致すること、`save_run`→`load_run` の往復でlogitsと共有が保たれることを `tmp/check_jp_char_gpt2.py` で確認した（2026-09-29）
- [x] 変換スクリプト `import_hf_gpt2.py`：`huggingface-hub` で `model.safetensors`・`config.json`・`vocab.json`・`merges.txt` を取得し、`JpCharGPT2` に重みを流し込み、語彙を `CharTokenizer` の形に変換して `runs/ku-nlp-gpt2-small-char/` に `save_run` する。`config.json` の層数・次元・ヘッド数・`context_length` はHF側の `config.json` から取る。依存に `safetensors` を足す。先に `CausalAttention` のmaskを `persistent=False` にして `model.pt` から外し（966MB → 362MB）、既存9runの `model.pt` からもmaskキーを除いて `load_run` とgreedy生成が変わらないことを確認した。変換後のパラメータ数は90,450,432（2026-09-29）
- [x] 一致確認（`tmp/check_import_hf_gpt2.py`、commitしない）：本物の `GPT2LMHeadModel` とhidden state用6文のlogitsの最大絶対差 2.3e-05。「日本の首都は」と6文の先頭文をpromptにしたgreedy 30文字の生成が一致。tokenizerは評価prompt10本と6文の計16本で比べ、Python実装の公式と同じ挙動（語彙にないtokenは `[UNK]`）のBPEと16本すべて一致した。改行を含むprompt 09・10は、Rust実装の公式（改行を黙って落とす）とだけ食い違う。`generate` で「日本の首都は」→「、東京都、神奈川県、埼玉県、千葉県、…」（2026-09-29）
- [x] 半角スペース・改行の扱いと `<s>` を付けるかを決めた。改行・半角スペースは `JpCharGPT2Tokenizer` で `encode` の前に取り除き、`<s>` は付けない（「作業の決め事」と `pretrained-model-plan.md` の「注意」、2026-09-29）
- [x] `generate`・`evaluate`・`visualize-attention`・`visualize-hidden-states`・`visualize-embeddings` を `--run-name ku-nlp-gpt2-small-char` で回した。全バッチvalidation loss 1.706350（モデルカードの eval loss 1.597 と桁が合う）。`visualize-hidden-states` は窓1,024で約4分（2026-09-29）
- [x] `plan.md` の「5.5 やらない。語彙もサイズも違うので読み込めない」を今回の決定に書き換え、「本から変える決定事項」に `JpCharGPT2`・変換スクリプト・`JpCharGPT2Tokenizer` を追記した。Stage 6を「学習済みモデルの読み込み」にし、付録D・第7章はStage 7の候補に繰り下げた（2026-09-29）
- [x] 評価（全バッチloss・固定10prompt）を `evaluation-results.md` の「学習済みモデル」に、可視化3種の観察を `visualization-results.md` の「学習済みモデル」に記録した。自作の `l12h12-d768-s5000-lr1e-3` との比較は読みにくくなるので載せない（2026-09-29）

## Stage 7：KVキャッシュ

`generate` にKVキャッシュを足し、生成が速くなることと結果が変わらないことを確かめる。設計と決めたことは `kv-cache-plan.md`。

- [x] `main.py generate` に生成時間と1文字あたりの時間を出す。`l4h4-d128-s5000-lr1e-3` は100文字 0.22秒（2.2ms/文字）、`ku-nlp-gpt2-small-char` は3.95秒（39.5ms/文字）。CPU、キャッシュなし（2026-09-29）
- [x] `kv_cache.py`（`HeadKVCache`・`KVCache`）と `CausalAttentionWithKVCache.forward(x, cache)`。ku-nlpの第1層ヘッド0で「日本の首都は」を空のキャッシュに一括で通した出力は元の `CausalAttention.forward` と完全一致（差0）、1文字ずつ通した出力との最大差 4.2e-07（float32の丸め誤差の範囲）。理解用の `CausalAttention`・`MultiHeadAttention`・`TransformerBlock` は変えていないので、`GPT` の `l4h4-d128-s5000-lr1e-3` は `evaluate` の全バッチloss 3.094512 も `visualize-attention` の出力も変わらず（2026-09-29）
- [x] `MultiHeadAttentionWithKVCache`・`TransformerBlockWithKVCache`・`JpCharGPT2` の配線。ku-nlpで「日本の首都は」の末尾logitsは一括と1文字ずつで最大差 1.2e-05（12層分の丸め誤差）、argmaxは同じ「、」。ku-nlpの `evaluate` の全バッチloss 1.706350 と生成サンプル20件は変わらず（2026-09-29）
- [x] `generate_with_kv_cache` と `main.py generate --kv-cache`。ku-nlpの固定10promptのgreedy 100文字は `generate` とtoken列が完全一致、ID04のsampling（T=0.8、seed 42〜44）も同seedで一致。prompt + `max_new_tokens` が `context_length` を超えると `ValueError`、`GPT` に使うと `TypeError`。`visualize-attention` の `attention_weights.json` は `l4h4-d128-s5000-lr1e-3` で変更前と同一（2026-09-29）
- [ ] `kv-cache-results.md`：ku-nlpの生成時間の比較、ku-nlpのキャッシュの大きさ、プロンプトキャッシュ・effort・有効期限の3つの問いへの考察。生成時間は100・200・300文字で記録済み（キャッシュなしは 35.9 → 56.7ms/文字と伸び、ありは約15ms/文字で一定。倍率 2.4 → 3.7倍、2026-09-29）。大きさと考察は未着手
- [ ] `progress.md`・`docs/glossary.md`・解説HTML `docs/demos/KVキャッシュで生成が速くなる理由.html` の「今の実装」を実装後に合わせる

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

### Stage 2：生成と損失（2026-09-25）

- 学習前の「日本の首都は」の続き（30文字）。本の "Featureiman Byeswickattribute argue" と同じく、訓練前はでたらめな文字列になる
  - T=0（greedy）：`盾!ワ銛剖般篠諱券倖貧†替凧酩語盛府柑肛ゲ敬↔登存Y厖岬翅巷`。2回呼んで同じ
  - T=0.8：`塗葯樫拠瘍校経ρカ啄ぎ七ơ蔓襠祐潔腰椒褪么混薙郡沈鯨鹸櫨滴様`、`ド胎数稔苔旧ぐ此銭句豪遅設拮瘡訂戯旗己馴劫怯聲憤渤髙爪噂⣦嫡`。呼ぶたびに違う
- 学習前の1バッチの損失：手計算（softmax→正解の確率→log→平均→-1倍）8.4707、`cross_entropy` 8.4707 で一致。`ln(4052)=8.307` よりやや大きい
  - 正解の文字に割り当てた確率は 0.0002〜0.0004 で、均等な 1/4052=0.000247 の前後に散らばっている
  - 均等より損失が大きいのは、初期値がランダムなので確率分布が均等ではなく、たまたま高い確率を付けた文字が正解になるとは限らないため。均等分布は「正解を知らないときの損失の下限」で、でたらめな偏りがあるぶん損失は上に出る
- パープレキシティ `exp(8.47)=4,773`。語彙数4,052より大きく、「次の文字の候補を全く絞れていない」状態

### Stage 3：訓練ループ（2026-09-25）

- 1層1ヘッド、2,000step、AdamW（lr 3e-4、weight_decay 0.1）。学習時間 43秒（評価11回込み）
- lossの推移（train は shuffle=False の先頭8バッチ、validation は先頭8バッチ）は `evaluation-results.md` の「l1h1 の学習曲線と全出力」節
- 2,000stepでも下がり続けている（最後の200stepで0.05）。パープレキシティは `exp(4.26)=71`
- validation lossがtrain lossより常にわずかに低い。過剰適合の逆で、先頭8バッチの文章の難しさの差と見られる（trainの先頭8バッチはたまたま難しい文章）。2本の差が広がらないことが大事で、差の符号は気にしない
- 「日本の首都は」のgreedy生成の変化
  - step 0: `敏歌霞榴酢米蛋泰措罷…`（でたらめ）
  - step 200: `、ののでののののののの…`（頻出文字「の」の繰り返し。読点を最初に出すのは学習した）
  - step 400: `、のです。  ーのでは、ので、のです。 している。 していた。`（「です。」「している。」の文末が出る）
  - step 1000: `、そのできることが、そのです。 そのできる。`（助詞と文末はそれらしいが、内容語が出ない）
  - step 1200〜2000: `、そのできるのです。 -  - 2000000000000…`（「0」の繰り返しに落ちる。greedyで一度「0」の次は「0」が最大になると抜けられない）
- 観察: 1層1ヘッドのgreedyでは、語句の切れ目と文末の形は覚えるが、話題を保った文章にはならない。「東京」は出ない。samplingでどうなるかは段階3の固定promptで見る

### Stage 3：保存と `generate` サブコマンド（2026-09-26）

モデル・Config・語彙を `runs/l1h1/` に保存し、`main.py generate` で読み込んで生成するようにした。読み込んだモデルの「日本の首都は」のgreedy生成が学習の最後（step 2000）の表示と一致し、保存→読み込みでモデルが変わらないことを確かめた。

- 観察: samplingにすると、greedyで落ち込んでいた「0」の繰り返しから抜け、「団体」「問題」「制動」など2文字の単語が現れる。一方で「生われる」「最気」のように、単語の途中で別の文字につながる箇所も多い。1文字1tokenなので、単語のつながりは2〜3文字先まで覚えられているが、文の意味は保てていない

### Stage 3：steps と学習率の見直し（2026-09-26）

2,000stepの基準（学習率3e-4）はlossが下がり続けたまま終わり、train と validation の差もなかったので、学習不足と判断して `steps` 5,000・学習率1e-3 で回した（`l1h1-s5000-lr1e-3`）。数値と全20出力は `evaluation-results.md` の「更新回数と学習率」節。

### Stage 3：d_model 64 → 128の比較（2026-09-26）

`config.py` の `d_model` だけを128に変え、`l1h1-d128-s5000-lr1e-3` として初期状態から学習した。比較元は `l1h1-s5000-lr1e-3`。数値と全20出力は `evaluation-results.md` の「幅 d_model」節。

### Stage 3：Token embeddingの地図（2026-09-26）

`l1h1-s5000-lr1e-3` の `tok_emb.weight` をPCAで2次元にし、自己完結HTML（検索・近傍一覧・頻度フィルタ付き）で観察した。数値と観察は `visualization-results.md`。

- PCAは自前実装（平均を引いて `torch.linalg.svd`）。2軸の寄与率は3.6%・3.3%で、64次元をほぼ均等に使っている（均等なら1軸1.6%）。d_model 128のrunで2.5%・2.1%、2,000stepのrunで3.2%・3.2%と、幅や学習量を変えても同じ
- PCAの図の近さは元の空間の近さを表さない。「高」と図で重なる態・制・心の類似度は−0.02〜0.13、類似度0.42の低は図で離れる。出現100回以上の全組で、2次元の距離と類似度の相関係数は−0.13
- 出現回数別の平均ノルムは1〜10回で4.84、11〜100回で4.95、101〜1,000回で5.24、1,001回以上で5.57。稀な文字の4.84は、初期値のノルム8にweight decayの倍率 (1−1e-4)^5000 ≈ 0.61 を掛けた値と一致し、ほぼ学習されていない。近傍候補を出現100回以上に絞る根拠
- 意味の近さ（長↔短、高↔低、東↔西、北・南）は元の空間の近傍に、表記の種類（ひらがな・漢字・カタカナ・数字）は2次元の配置に現れた
- PCAの図では近傍が隣に来ないので、umap-learn 0.5.12を足してUMAP（metric cosine）に切り替えられるようにした。設定は「コサイン上位10の文字が図の最近傍10に入る個数」の全文字平均で選び、n_neighbors 5・min_dist 0にした（既定の15・0.1で1.4個、採用値で2.2個。偶然なら0.06個、PCAの図では0.19個）
- UMAPは全体の一致数がseedで安定する一方、特定の組が図で何番目に近いかはseedで大きく変わる（高→低が603位・5位・5位）。図の隣接だけで「近い」と言わず、近傍一覧の類似度で確かめる

### Stage 3：Hidden stateの変化（2026-09-26）

`l1h1-s5000-lr1e-3` で、前文の違う6文の「行」と「の」について層ごとのhidden stateを観察した。数値と観察は `visualization-results.md` の「基準モデル（1層・64次元）」節。

### Stage 4：OneHeadGPT と1層の一致確認（2026-09-27）

`l1h1-d128-s5000-lr1e-3` の重みをキー名を付け替えて `n_layers=1` の `OneHeadGPT` に流し込むと、全バッチvalidation lossが保存済みの値（3.376540）と一致した。`OneHeadGPT` の1層は `OneLayerOneHeadGPT` と同じ計算なので、1層は学習し直さない（`tmp/check_gpt_model.py`、commitしない）。既存3runの `config.json` には `"n_layers": 1` を手で足した。

### Stage 4：4層の学習（2026-09-27）

`config.py` の `n_layers` を4にして `l4h1-d128-s5000-lr1e-3` を学習した。比較元は1層の `l1h1-d128-s5000-lr1e-3`。数値と全20出力は `evaluation-results.md` の「層数」節。

### Stage 4：1層のstepを増やせば4層に届くか（2026-09-27）

参考実験。「4層の差はstep不足では」を確かめるため、1層・128次元を20,000stepで学習した（`l1h1-d128-s20000-lr1e-3`）。事前の「等比で縮むなら3.20止まり」の予想は外れ、減り幅は等比より遅く縮む。数値と全20出力は `evaluation-results.md` の「層数」節。

### Stage 4：4層のhidden stateと128次元のembedding（2026-09-27）

`l4h1-d128-s5000-lr1e-3` で層ごとのhidden stateを、`l1h1-d128-s5000-lr1e-3` と合わせて128次元のembeddingを観察した。観察は `visualization-results.md` の「層数」と「幅 d_model」の節。

### Stage 4：学習をMac GPU（MPS）に移す（2026-09-27）

4層の学習が384秒かかるようになったので、学習だけMPSに移した。`tmp/bench_mps.py`（commitしない）で計測してから決めた。

- 学習1stepは1層で30.4ms → 12.2ms、4層で66.5ms → 25.1ms（約2.5倍）。バッチの `.to("mps")` 転送は0.4ms/stepで無視できる
- 1文字ずつの生成は逆に遅い（1層50文字で0.018秒 → 0.13秒）。evaluateは全バッチlossが速くなる分を20本の生成が食いつぶし、合計では速くならない。UMAPは `.numpy()` がMPSで落ち、`torch.linalg.svd` はCPUにフォールバックする。生成・評価・可視化はCPUに残した
- `torch.set_default_device("mps")` を入口で1回呼ぶ案は不採用。`DataLoader(shuffle=True)` の乱数生成器がCPUのまま `randperm` の出力先だけMPSになり例外になる。Datasetの2万個の小さいテンソルもGPUに作られ、データ準備が1.5秒 → 15秒になる
- 1層をMPSで5,000step学習した `l1h1-d128-s5000-lr1e-3-mps` は、CPUの `l1h1-d128-s5000-lr1e-3` と全バッチvalidation loss 3.376540・学習中の最終値 3.279632・greedy生成がすべて一致した。初期値とshuffle順はCPUの乱数で作ってからGPUへ送るので乱数の消費が変わらず、Dropoutもないため。学習時間は163.2秒 → 64.3秒

### Stage 5：Multi-head Attention の作り方の選択（2026-09-27）

本の3.6には、`CausalAttention` を並べて `torch.cat` する `MultiHeadAttentionWrapper`（3.6.1）と、1つの `W_query` を `view`・`transpose` でヘッドに分割する `MultiHeadAttention`（3.6.2）の2つがある。最初は両方を作ったが、構造を理解する目的では並べる形の方が読みやすいので、並べる形に `out_proj` を足したものを `MultiHeadAttention` として残し、分割方式は消した。dropoutはなし。`out_proj` は本と同じくbiasあり。

両方があった時点で確かめたこと。

- 分割方式の `W_query.weight`（shape `[128, 128]`）を行方向に64ずつ切って並べる形の各ヘッドに入れ、`out_proj` を単位行列・bias 0 にすると、2つの出力は完全一致（最大差 0.0）。「1つの行列を分割する」と「小さな行列を並べて連結する」は同じ計算
- 速度（B=16・T=256・D=128、並べる形にも `out_proj` 相当のLinearを足して比較）。4ヘッドのforward+backwardはCPUで約7ms対約7ms、MPSで1.7ms対2.0msと、この規模では分割方式は速くない。MPSではむしろ `contiguous()` のコピー分だけ遅く、8ヘッドで2割ほど開いた
- GPT-2 small構成（12層・12ヘッド・768次元、B=16・T=256）でも差はない（2026-09-28、`tmp/bench_mha_gpt2.py`、commitしない）。乱数バッチで200stepの学習を回すと、並べる形が167.9秒（1step 839ms）、分割方式が169.9秒（1step 850ms）で1%程度の差。この規模では1stepの大半をFeedForwardと出力層の行列積が占め、Attentionの呼び出し回数を減らしても全体はほとんど縮まない。本の言う「効率的」はMPSでは規模を問わず効かず、並べる形のままで十分

残した `MultiHeadAttention` は、`n_heads=1` で `out_proj` を無効化すると `CausalAttention` と完全一致し、パラメータ数は1・2・4ヘッドで変わらない（`tmp/check_multihead.py`、commitしない）。

### Stage 5：4層で1ヘッドと4ヘッドの比較（2026-09-28）

`config.py` を `n_layers=4` にして `GPT` を `n_heads=1` と `n_heads=4` で学習した。run名は、`OneHeadGPT` の4層run `l4h1-d128-s5000-lr1e-3` と区別するため、1ヘッドの `GPT` に `-outproj` を付けた。数値と全20出力は `evaluation-results.md` の「ヘッド数」節。

### Stage 6の下見：GPT-2 smallの形で学習（2026-09-28）

`config.py` を `d_model=768`・`n_layers=12`・`n_heads=12` にして `l12h12-d768-s5000-lr1e-3` を学習した。数値と全20出力は `evaluation-results.md` の「モデル規模」節。

- 事前に `tmp/bench_gpt2_small.py`（commitしない）で乱数バッチの1stepを測り、5,000stepで約71分と見積もってから回した。`model.pt` は404MB
- 学習中の保存は `train_model` にないので、validationが最良だったstep 4400付近のモデルは残っていない。Stage 6でモデルを大きくするなら、validation最良時点の別保存（`runs/<run_name>/best/`）とdropout、またはデータ増を検討する。データを足すと語彙が変わって既存runと比べられなくなるので、別ディレクトリ（`data/fineweb-japanese-20k/` など）で別の比較群にする
- 75分の学習はBashのバックグラウンド実行（上限10分）では打ち切られる可能性があるので、`nohup` で切り離して `tmp/train-l12h12.log` に書き、`tail -f` で監視した

### Stage 6の下見：12層モデルでは「近い文脈」が読めず、コーパス拡大と3分類で直した（2026-09-28）

`l12h12-d768-s5000-lr1e-3` の最終層の「近い文脈」は、銀行2件のあとに現金・日本などが当時の偶然の水準（全位置の上位0.1%点、0.52）をわずかに超える0.55〜0.65で並び、読めなかった。原因は2つ（`tmp/check_neighbor_tail.py`、commitしない）。

- 先頭39,936位置に銀行・旅行は2回、実行・発行は1回しか出ない（validation全体では55・60・58・29回）
- 先頭39,936位置では、12層モデルのLayer 1の時点で、銀行以外の「行」との類似度が0.59以下に落ちる。12層モデルの「行」は、「行という文字」ではなく「銀行という単語の末尾」を表している。4層モデルのLayer 1は別の熟語の「行」も0.6〜0.7で近く、上位10が「行」で埋まっていたので読めていた

コーパスをvalidation全体657,408位置に広げると、今度は上位10が全部「銀行」になり、単語を認識している以上のことが読めない。そこで近傍を「同じ2文字（銀行の行、55件）」「同じ文字・前が違う（旅行・走行の行、1,132件）」「別の文字（656,221件）」の3列に分けた（`NEIGHBOR_GROUPS`）。12層モデルのLayer 12で、銀行の「行」には、別の熟語の「行」（国内旅行0.63・走行0.57）と預金・融資・金額の末尾（0.60〜0.67）が同じくらいの類似度で並ぶ。「別の文字」は候補が656,221件と他の列の数百倍あり、候補が多いほど上位の値は高く出るので、列どうしの上位の値をそのまま比べない。自作12層の観察は `visualization-results.md` には載せず、学習済みモデルの観察を「学習済みモデル」の節に書いた。

- 全層のベクトルを溜めると13層×657k×768次元で約26GBになるので、窓ごとにforwardして上位だけを残す `search_neighbors` にした。旧方式と同じ39,936位置で比べ、Layer 1以降の上位は全体・分類ごととも一致（`tmp/check_search_neighbors.py`、commitしない）。Layer 0は同じ文字が同じ窓内位置にあるとベクトルが同一になり、同点の並びだけ変わる
- 実行時間は4層で15秒、12層で3分22秒（CPU）

### Stage 6：日本語GPT-2の学習済みモデルの調査（2026-09-29）

Hugging Face上でGPT-2アーキテクチャ（`GPT2LMHeadModel`）の日本語モデルを調べ、`ku-nlp/gpt2-small-japanese-char` を選んだ。

- 1文字単位のモデルは `ku-nlp/gpt2-{small,medium,large}-japanese-char` だけ。他はSentencePiece（rinna・abeja・colorfulscoop）、6万語彙のBPE（ClassCat）、Juman++の分かち書き前提（nlp-waseda）で、サブワード単位
- ku-nlp smallは12層・768次元・12ヘッド・位置埋め込み1,024・語彙6,000・`gelu_new`（tanh近似）・LayerNorm eps 1e-5。Wikipedia + CC-100 + OSCARの171GBをA100 1枚で約3か月。`model.safetensors` は374MB
- 自作 `GPT` との差は4点。Q/K/Vにbiasがある、Q/K/Vが1つの `c_attn`（768×2,304）に全ヘッド分連結されている、出力ヘッドがトークン埋め込みと同じ行列、位置埋め込みが1,024行。GELU・LayerNorm・Pre-LayerNorm・ショートカット接続の順序は同じ。dropout（0.1）は推論では無効
- tokenizerは公式の `vocab.json`・`merges.txt` が付属する。「日本の首都は東京です。」は11文字→11token、「鬱蒼とした森で薔薇が咲く」も12文字→12token。`vocab.json` のキーはUTF-8のバイトを表示用の文字に置き換えた形（「日」は `æĹ¥`）で、文字に戻す復号が要る。半角スペースはtokenにならず消える（モデルカードの「全角スペースを使う」注意はこのため）
- `config.json` の `eos_token_id` は2だが、`vocab.json` ではID 2が `<s>`、3が `</s>`。文書区切りには `</s>` を使い、生成結果で確かめる

### Stage 6：改行・半角スペースは除き、`<s>` は付けない（2026-09-29）

除くとlossは2.527から1.635に下がる。`<s>` は同じ予測位置で比べると改善が0.01程度で、付けると可視化の対象位置がずれるので付けない。validation先頭200文書を文書ごとに単独で測った（`tmp/check_bos_unk.py`・`tmp/check_bos_same_positions.py`、commitしない）。

| 改行・半角スペース | `<s>` なし | `<s>` あり |
|---|---:|---:|
| 残す（`<|unk|>` になる） | 2.5265 | 2.5352 |
| 除く | 1.6354 | 1.6346 |

- 表の `<s>` ありは `<s>` から1文字目を当てる位置（平均loss 6.6）も含む。同じ予測位置だけで比べると、除く条件で `<s>` ありは0.009低い（188/200文書で改善）
- 残す条件では、正解が `<|unk|>` の位置（tokenの2.3%）が平均15.4、その直後（2.2%）が平均29.9で、この2つでlossの40%を占める。残りの位置は平均1.589
- 改行・半角スペース以外で `<|unk|>` になる文字はvalidation全体で54個
- `<s>` の有無で生成は変わるが質の優劣はない。「日本の首都は」の次は、なしで「、」0.13・「東」0.09、ありで「東」0.17・「、」0.13
- `<|unk|>` が文書の境界として学習されている根拠は、`</s> <|unk|>` の次に `<s>` を1.000で予測すること。学習データで常に同じ並びだったtokenは、違う文脈でも確率1.000で同じ続きを出す。「なぜ自信満々で間違えるのか」の手がかりになる

### Stage 6：学習済みモデルの評価と可視化の記録（2026-09-29）

数値と観察は `evaluation-results.md` の「学習済みモデル」と `visualization-results.md` の「学習済みモデル」。記録のために分かった作業上のこと。

- Attentionの「先頭」が均等（0.15）を大きく超えるときは、先頭位置の値ベクトルのノルムと残差ストリームのノルムを見る。学習済みモデルは先頭位置だけ残差ストリームのノルムが約730（他は48〜140）で、先頭を見るヘッドの値ベクトルは他の2〜8%。`tmp/check_attention_sink.py`・`tmp/check_attention_sink2.py`（commitしない）で、位置ごとの先頭重み・値ベクトルとキーのノルム・`<s>` を付けた場合・先頭の文字を替えた場合を測った
- `visualize-attention` のヒートマップは、先頭に集まるモデルでは先頭列だけが濃くなり残りが読めない。先頭列を除いて色を付け直す表示は未実装
- hidden stateの「偶然の水準」は学習済みモデルでは層が深いほど上がる（Layer 6で0.70、Layer 12で0.89〜0.94）。後半の層の類似度は偶然の水準との差で読む
- `hidden_states.json`・`embedding_map.json`・`attention_weights.json` の集計は `tmp/summarize_hidden_states.py`・`tmp/summarize_embedding_map.py`（commitしない）で行い、JSONそのものは読まない。`embedding_map.json` には近傍が入っておらず、HTML側でvectorから計算しているので、記録用の近傍はスクリプトで計算した
