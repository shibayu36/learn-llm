# llms-from-scratch の進捗

計画は `plan.md`。ここでは各Stageの作業状態と、実行して分かったことを記録する。チェックは実行して確認できたときに付ける。

## 現在の作業

Stage 5（Multi-head Attention）を進行中（2026-09-28）。4層・128次元で1ヘッド（`l4h1-d128-s5000-lr1e-3-outproj`）と4ヘッド（`l4h4-d128-s5000-lr1e-3`）を学習・評価し、`evaluation-results.md` の「ヘッド数」に記録した。次は4ヘッドのrunでヘッドごとのAttention weightを観察する。Stage 6の下見としてGPT-2 smallの形（`l12h12-d768-s5000-lr1e-3`）も学習し、`evaluation-results.md` の「モデル規模」に記録した。`config.py` は `d_model=128`・`n_layers=4`・`n_heads=4`・`steps=5000` に戻してある。固定10promptの学習前出力は未記録。

## 作業の決め事

- `evaluation-results.md` は「変えた条件のカテゴリ（層数・幅・更新回数と学習率）を `##`、問いへの答えが読める見出しを `###`」の構成にする。各節は結論の段落、設定の段落、詳細の順に書く。新しいカテゴリ・節は上に足す。各runの学習曲線と全20出力は末尾にまとめず、そのrunを最初に使った問いの節の末尾に置く。計画外の参考実験は1段落に収める
- `visualization-results.md` も同じ構成にする。変えた条件（層数・幅）を `##`、「変えたら内部表現がどう変わったか」を `###` にし、1つの節でhidden stateとtoken embeddingの両方を使う。条件を変える前の1モデルだけの観察は「基準モデル」のカテゴリに置く。見出しに日付は書かない（別の日にやっても条件は変わらない）。節の見出しは結論が読めるように書き、詳細な表は見出しと最初の段落のあとに置く
- `runs/` をcommitしないのは、model.pt を入れないなら config.json や metrics.json だけ残しても再現できないため
- `load_run` は復元だけを行い、`model.eval()` は生成・評価する側で呼ぶ。読み込んだモデルを何に使うかは呼び出し側が決めることなので
- `generate.py` のロジットの説明は、学習後のモデルに「日本の首都は」を入れて観察した実値の表にした。観察に使ったスクリプトは `tmp/show_logits.py`（commitしない）
- 保存・読み込みは `train.py` ではなく `checkpoint.py` に分けた。`generate` が `train.py` を読み込むと「生成に訓練が要る」ように見えるため
- 新しい用語が出る節では用語集 `docs/glossary.md` に追記する
- 学習だけ `Config.device` の装置（既定 `"mps"`）で行い、生成・評価・可視化はCPUのまま。モデルと `GPTDataset` のID列を最初から `config.device` に置き、DataLoaderがその装置上のバッチを返すようにする。本のように `calc_loss_batch` などへ `device` を引き回さないため。`device` は `config.json` に保存しない
- optimizerは本と同じく `train_model` の外で作って渡す形のままにする。モデルとoptimizerは `.grad` を通して暗黙につながるので、optimizerは必ずそのモデルの `parameters()` で作る。Stage 6でスケジューラやパラメータのグループ分けを入れるときも外で作る

## Stage 0：部品の実装

- [x] 日本語データ1万文書を取得し、train/validationに分けて固定する（`prepare_data.py`、2026-09-25）
- [x] `Config`（d_model 64、context_length 256、batch 16、steps 2000。context_lengthは64から256に変更、2026-09-25）
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
- lossの推移（train は shuffle=False の先頭8バッチ、validation は先頭8バッチ）

  | step | train | validation |
  |---:|---:|---:|
  | 0 | 8.464 | 8.462 |
  | 200 | 5.785 | 5.687 |
  | 400 | 5.541 | 5.436 |
  | 600 | 5.239 | 5.120 |
  | 800 | 4.968 | 4.851 |
  | 1000 | 4.776 | 4.662 |
  | 1200 | 4.645 | 4.533 |
  | 1400 | 4.549 | 4.439 |
  | 1600 | 4.476 | 4.367 |
  | 1800 | 4.417 | 4.311 |
  | 2000 | 4.368 | 4.264 |

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

- `uv run main.py train` を再実行すると、lossの推移も学習中の生成も前回（2026-09-25）と1文字違わず同じだった。seed固定でDataLoaderの順番とモデルの初期値が決まるので、同じコードなら同じ結果になる
- `runs/l1h1/` に3ファイル。`model.pt` は2.59MB（パラメータ580,800個 × float32の4バイト = 2.32MB に、名前などの情報が乗る）、`vocab.json` は28KB、`config.json` は221バイト
- `uv run main.py generate "日本の首都は"` の結果
  - T=0：`、そのできるのです。 -  - 1900000000000000000000000010001000`。2回とも同じで、学習の最後（step 2000）の表示とも同じ。保存→読み込みでモデルが変わっていないことの確認になる
  - T=0.8：`2005196ノース団ル事にようとが行っていたよりでしても問題 15.....月代としていといる。`、`言うこの制動を生われる。 といんだ度団体は、このはみな、最気いよりします。 大職や資ばなど配の記来の`。2回とも違う
- 観察: samplingにすると、greedyで落ち込んでいた「0」の繰り返しから抜け、「団体」「問題」「制動」など2文字の単語が現れる。一方で「生われる」「最気」のように、単語の途中で別の文字につながる箇所も多い。1文字1tokenなので、単語のつながりは2〜3文字先まで覚えられているが、文の意味は保てていない

### Stage 3：steps と学習率の見直し（2026-09-26）

2,000stepの基準（学習率3e-4）はlossが下がり続けたまま終わり、train と validation の差もなかったので、学習不足と判断して `steps` 5,000・学習率1e-3 で回した。詳細は `evaluation-results.md` の `l1h1-s5000-lr1e-3`。

- validation loss は 4.264 → 3.649、パープレキシティは 71 → 38。学習時間105秒
- 学習率の効果と step の効果は分けて読める。同じ2,000step時点で 4.264 → 3.931 が学習率、そこから5,000stepで 3.649 が step。学習率1e-3でも序盤に跳ねなかった
- trainを約3.5周しても train と validation の差は約0.08で一定。過剰適合の気配はなく、データはまだ余っている
- lossが4.3から3.6に下がると、greedyは「記号の繰り返し」から「助詞と定型句でつないだ文らしいもの」に変わる。ただし同じ句を繰り返し、「東京」は出ない

### Stage 3：d_model 64 → 128の比較（2026-09-26）

`config.py` の `d_model` だけを128に変え、`l1h1-d128-s5000-lr1e-3` として初期状態から学習した。比較元は `l1h1-s5000-lr1e-3`。batch 16・5,000step・学習率0.001など他の設定は同じ。batch sizeの変更実験は行っていない。

- 全バッチvalidation lossは3.717512 → 3.376540（約9.2%低下）。評価対象は両方とも657,408token
- パラメータ数は580,800 → 1,251,712（約2.16倍）。学習時間は111.8秒 → 163.2秒（約1.46倍、定期評価・生成を含む）
- samplingでは「確認していくことが必要です」などの語句が見られるが、プログラミングの学び方を説明できていない。greedyは全10promptが「そのため」の反復に陥り、lossの改善と生成文の質は一致しなかった

この結果を踏まえ、Stage 4・5の幅は128に決めた（2026-09-27）。

### Stage 3：Token embeddingの地図（2026-09-26）

`l1h1-s5000-lr1e-3` の `tok_emb.weight` をPCAで2次元にし、自己完結HTML（検索・近傍一覧・頻度フィルタ付き）で観察した。数値と観察は `visualization-results.md`。

- PCAは自前実装（平均を引いて `torch.linalg.svd`）。2軸の寄与率は3.6%・3.3%で、64次元をほぼ均等に使っている（均等なら1軸1.6%）。d_model 128のrunで2.5%・2.1%、2,000stepのrunで3.2%・3.2%と、幅や学習量を変えても同じ
- PCAの図の近さは元の空間の近さを表さない。「高」と図で重なる態・制・心の類似度は−0.02〜0.13、類似度0.42の低は図で離れる。出現100回以上の全組で、2次元の距離と類似度の相関係数は−0.13
- 出現回数別の平均ノルムは1〜10回で4.84、11〜100回で4.95、101〜1,000回で5.24、1,001回以上で5.57。稀な文字の4.84は、初期値のノルム8にweight decayの倍率 (1−1e-4)^5000 ≈ 0.61 を掛けた値と一致し、ほぼ学習されていない。近傍候補を出現100回以上に絞る根拠
- 意味の近さ（長↔短、高↔低、東↔西、北・南）は元の空間の近傍に、表記の種類（ひらがな・漢字・カタカナ・数字）は2次元の配置に現れた
- PCAの図では近傍が隣に来ないので、umap-learn 0.5.12を足してUMAP（metric cosine）に切り替えられるようにした。設定は「コサイン上位10の文字が図の最近傍10に入る個数」の全文字平均で選び、n_neighbors 5・min_dist 0にした（既定の15・0.1で1.4個、採用値で2.2個。偶然なら0.06個、PCAの図では0.19個）
- UMAPは全体の一致数がseedで安定する一方、特定の組が図で何番目に近いかはseedで大きく変わる（高→低が603位・5位・5位）。図の隣接だけで「近い」と言わず、近傍一覧の類似度で確かめる

### Stage 3：Hidden stateの変化（2026-09-26）

`l1h1-s5000-lr1e-3` で、前文の違う6文の「行」（位置9）と「の」（位置10）について、Layer 0とLayer 1のhidden stateを `register_forward_hook` で取り出し、層ごとの「次の文字の予測（logit lens）」と「validation先頭4万位置の中で近い文脈」の表をHTMLにした。数値と観察は `visualization-results.md`。

- hidden stateを `final_norm` → `out_head` に通すと途中の層でも次の文字の予測として読める。近い文脈は、同じ層のvalidationのベクトルとのコサイン類似度で、embedding地図の近傍一覧の文脈版にあたる
- 「行」はLayer 1で、銀行・旅行のあとは動詞の行（行う・行った）の仲間、実行・発行のあとは熟語の行（銀行・非行・先行）の仲間になった。前文と直前の単語を入れ替えた文を足すと両方が効いており、直前の1文字の方が動かす量は大きかった
- 「の」は6文とも予測が平坦（最大0.02）で近傍も名詞に続く「の」ばかりになり、文脈でほとんど変わらない
- 予測上位に前文の末尾の文字（楽しもう→う、受領した→し）が入る。ヘッドが前の文字を写している可能性があり、Attentionの重みを見るときに確かめる
### Stage 4：OneHeadGPT と1層の一致確認（2026-09-27）

`Config.n_layers`（既定1）を足し、`TransformerBlock` を `nn.Sequential` に `n_layers` 個並べた `OneHeadGPT` を `gpt.py` に追加した。`main.py train` は `OneHeadGPT` で学習し、`checkpoint.load_run` はクラス名で `OneLayerOneHeadGPT` と `OneHeadGPT` を組み立て分ける。既存3runのローカルの `config.json` には `"n_layers": 1` を手で足した。

`tmp/check_gpt_model.py`（commitしない）で `n_layers=1` の `OneHeadGPT` が `OneLayerOneHeadGPT` と同じものであることを確かめた。

- 同じseed（42）で作ると、パラメータ数1,251,712・17個のパラメータの初期値がすべて一致。パラメータの作られる順番が同じなので乱数の消費も同じになる。`state_dict` のキーは `trf_block.` と `trf_blocks.0.` で違うだけ
- ランダムな入力 `[2, 16]` のlogitsが完全一致（`torch.equal`）。cross entropyで `backward` した勾配も全パラメータで完全一致。forwardとbackwardが同じなら学習の経過も同じになるので、1層は学習し直さない
- `l1h1-d128-s5000-lr1e-3` の重みをキー名を付け替えて `OneHeadGPT` に流し込み、全バッチvalidation lossを測ると 3.3765399842247414 で、保存済みの値と小数点以下すべて一致。評価token数も657,408で同じ

### Stage 4：4層の学習（2026-09-27）

`config.py` の `n_layers` を4にして `l4h1-d128-s5000-lr1e-3` を学習した。比較元は1層の `l1h1-d128-s5000-lr1e-3`。数値と全20出力は `evaluation-results.md`。

- パラメータ数 1,251,712 → 1,795,840（1層あたり181,376増）。学習時間 163.2秒 → 383.9秒（約2.35倍）、評価時間 3.0秒 → 7.6秒
- 全バッチvalidation loss 3.376540 → 3.067810（約9.1%低下）。先頭8バッチのvalidation lossは200step時点から4層が低く、差は終盤で約0.3。5,000stepでもまだ下がり続けている
- trainとvalidationの差は約0.04〜0.1で一定。パラメータが1.4倍になっても過剰適合の気配はない
- 幅を倍にした効果（-0.34、+670,912パラメータ）と層を4倍にした効果（-0.31、+544,128パラメータ）はlossの低下量が近い
- greedyは「そのため、」だけの反復から抜け、「その後の人々があると思います。」のような文末まで届く句が出る。反復の単位は句から文に伸びたが消えてはいない。「学校の学校の学校」のような短い語の反復は新たに出た
- samplingでは「プログラミングを学ぶには、」の10件中9件にIT関連語が出る（1層は4件程度）。lossの改善が「promptの話題に関係する語彙を選ぶ」ところまで届いた
- 学習中のstep 2400で初めて「東京都」が出た（1層は5,000stepまで出ず）。ただし最終モデルのgreedyには出ない

### Stage 4：1層のstepを増やせば4層に届くか（2026-09-27）

参考実験。「4層の差はstep不足では」を確かめるため、1層・128次元を20,000stepで学習した（`l1h1-d128-s20000-lr1e-3`、677.2秒）。全バッチvalidation lossは3.113614で4層5,000stepの3.067810に届かず、trainを約14周しても過剰適合せず、greedyの反復も変わらなかった。事前の「等比で縮むなら3.20止まり」の予想は外れ、減り幅は等比より遅く縮む。

### Stage 4：4層のhidden stateと128次元のembedding（2026-09-27）

`visualize-hidden-states` と `visualize-embeddings` は `model.modules()` から `TransformerBlock` を拾うので、4層の `OneHeadGPT` にコード変更なしで動いた。HTMLの層ボタンもLayer 0〜4で描画される。観察は `visualization-results.md` の「層数」と「幅 d_model」の節。

- 埋め込みの近傍を読むときは偶然の水準を先に出す。ランダムな1,632本のベクトルで近傍1位の平均は64次元で0.41、128次元で0.30。実際の埋め込みの平均と同じなので、上位10の大半は偶然
- 近傍の理由は、訓練データの前後1文字の出現分布の類似度と、`out_head` の行どうしの類似度で確かめた（`tmp/why_neighbors.py`、commitしない）。`tok_emb` と `out_head` は別の行列で、持っている情報が違う

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
- 速度（B=16・T=256・D=128、並べる形にも `out_proj` 相当のLinearを足して比較）。4ヘッドのforward+backwardはCPUで約7ms対約7ms、MPSで1.7ms対2.0msと、この規模では分割方式は速くない。MPSではむしろ `contiguous()` のコピー分だけ遅く、8ヘッドで2割ほど開いた。本の言う「効率的」は、GPT-2の12ヘッド・768次元のように行列が大きいときの話

残した `MultiHeadAttention` について `tmp/check_multihead.py`（commitしない）で確かめたこと。

- 4ヘッドの出力は `[2, 16, 128]`、各ヘッドの出力は `[2, 16, 32]`
- `n_heads=1` は `out_proj` を無効化すると `CausalAttention` と完全一致
- パラメータ数は1・2・4ヘッドとも 65,664 で変わらない。`CausalAttention` の 49,152 との差 16,512 は `out_proj`（128×128 + bias 128）
- `d_out=128` を3ヘッドにしようとすると `ValueError`

### Stage 5：4層で1ヘッドと4ヘッドの比較（2026-09-28）

`config.py` を `n_layers=4` にして `GPT` を `n_heads=1` と `n_heads=4` で学習した。run名は、`OneHeadGPT` の4層run `l4h1-d128-s5000-lr1e-3` と区別するため、1ヘッドの `GPT` に `-outproj` を付けた。数値と全20出力は `evaluation-results.md` の「ヘッド数」。

- パラメータ数は1ヘッドも4ヘッドも1,861,888で同じ。`OneHeadGPT` の4層より `out_proj` 4層分の66,048多い
- 全バッチvalidation lossは `OneHeadGPT` 3.067810、1ヘッドの `GPT` 3.107401、4ヘッドの `GPT` 3.094512。ヘッド数の差は0.013で、層数（0.31）・幅（0.34）の効果より1桁以上小さい。`out_proj` を足すと0.04悪くなった。学習seedを変えたばらつきを測っていないので、どちらの差も効果とは読まない
- 先頭8バッチの曲線は3runともstep 800まで重なる。`OneHeadGPT` はstep 1000から先に下がり、`GPT` の2runは4000stepまで1ヘッドがわずかに低く、4200step以降で4ヘッドが下回る
- 学習時間はMPSで1ヘッド142.3秒、4ヘッド170.1秒。ヘッドを順に計算するので呼び出し回数が増える分
- 生成は反復の単位が変わる（「そのため、そのためには」→「それは、それぞれの人が」→「学習塾の学習」・引用符「」）が、文をまたいで意味がつながらないのは3runとも同じ。4ヘッドのgreedyはID04で「学ぶ」を拾って「学習塾」で埋めた。samplingのIT関連語は9件・6件・4件と減った
- 同じseedのsamplingは別のモデルでも書き出しが一致しやすい（seed 42「癒している。」など）。乱数の列が同じなので、候補の確率の付け方が似ている間は同じ文字が選ばれる
- ヘッドを分けた効果はlossでは見えないので、次のタスクでヘッドごとのAttention weightを見て確かめる

### Stage 6の下見：GPT-2 smallの形で学習（2026-09-28）

`config.py` を `d_model=768`・`n_layers=12`・`n_heads=12` にして `l12h12-d768-s5000-lr1e-3` を学習した。context 256・batch 16・5,000step・学習率0.001・語彙4,052は同じ。数値と全20出力は `evaluation-results.md` の「モデル規模」。

- 事前に `tmp/bench_gpt2_small.py`（commitしない）で乱数バッチの1stepを測り、0.85秒・GPUメモリ1.5GB・5,000stepで約71分と見積もってから回した。実際の学習時間は4,632.3秒（77分、評価・生成込み）。`model.pt` は404MB
- パラメータ数91,448,832（4層4ヘッドの約49倍）。全バッチvalidation lossは3.094512 → 2.681605。評価はCPUで210.5秒（4層は10.5秒）
- 学習率0.001・warmupなしでも序盤に跳ねなかった
- 初めて過剰適合が見えた。trainは1,447stepで1周（23,163窓 ÷ batch 16、端数切り捨て）、5,000stepで約3.5周。3周目に入るstep 3000と4周目のstep 4400で、trainの先頭8バッチのlossだけ段差で下がり、validationは止まる。step 4600でvalidationが初めて上がり（2.604 → 2.619）、最終の差は0.22。4層以下では3.5周しても差が開かなかった
- 過剰適合してもtrainの文書をそのまま吐きはしない。trainの先頭3文書の先頭40文字からgreedyで続きを生成すると、実際の続きとの先頭一致は0〜1文字で4層と同じ（`tmp/check_memorization.py`、commitしない）。差0.22は文章の丸暗記ではなく、trainに出た語の組み合わせを覚える方向に出ている
- 生成はgreedyの10promptすべてが文法的に完結した文になり、「健康を保つためには、健康に役立てています」「英語を話すことができます。英語教室」とpromptの語を保って続ける。一方、「健康的な健康的な」「英会話\n英会話」の短い反復と、文をまたぐと同じことを言い直す問題は残る。「日本の首都は」に東京は出ない
- 学習中の保存は `train_model` にないので、validationが最良だったstep 4400付近のモデルは残っていない。Stage 6でモデルを大きくするなら、validation最良時点の別保存（`runs/<run_name>/best/`）とdropout、またはデータ増を検討する。データを足すと語彙が変わって既存runと比べられなくなるので、別ディレクトリ（`data/fineweb-japanese-20k/` など）で別の比較群にする
- 75分の学習はBashのバックグラウンド実行（上限10分）では打ち切られる可能性があるので、`nohup` で切り離して `tmp/train-l12h12.log` に書き、`tail -f` で監視した
