# llms-from-scratch の進捗

計画は `plan.md`。ここでは各Stageの作業状態と、実行して分かったことを記録する。チェックは実行して確認できたときに付ける。

## 現在の作業

Stage 5（Multi-head Attention）を進行中（2026-09-28）。4層・128次元で1ヘッド（`l4h1-d128-s5000-lr1e-3-outproj`）と4ヘッド（`l4h4-d128-s5000-lr1e-3`）を学習・評価し、`evaluation-results.md` の「ヘッド数」に記録した。次は4ヘッドのrunでヘッドごとのAttention weightを観察する。Stage 6の下見としてGPT-2 smallの形（`l12h12-d768-s5000-lr1e-3`）も学習し、`evaluation-results.md` の「モデル規模」に記録した。`config.py` は `d_model=128`・`n_layers=4`・`n_heads=4`・`steps=5000` に戻してある。固定10promptの学習前出力は未記録。

## 作業の決め事

- `evaluation-results.md` は「変えた条件のカテゴリ（層数・幅・更新回数と学習率）を `##`、問いへの答えが読める見出しを `###`」の構成にする。各節は結論の段落、設定の段落、詳細の順に書く。新しいカテゴリ・節は上に足す。各runの学習曲線と全20出力は末尾にまとめず、そのrunを最初に使った問いの節の末尾に置く。計画外の参考実験は1段落に収める
- `visualization-results.md` も同じ構成にする。変えた条件（層数・幅）を `##`、「変えたら内部表現がどう変わったか」を `###` にし、1つの節でhidden stateとtoken embeddingの両方を使う。条件を変える前の1モデルだけの観察は「基準モデル」のカテゴリに置く。見出しに日付は書かない（別の日にやっても条件は変わらない）。節の見出しは結論が読めるように書き、詳細な表は見出しと最初の段落のあとに置く
- `runs/` をcommitしないのは、model.pt を入れないなら config.json や metrics.json だけ残しても再現できないため
- `load_run` は復元だけを行い、`model.eval()` は生成・評価する側で呼ぶ。読み込んだモデルを何に使うかは呼び出し側が決めることなので
- 保存・読み込みは `train.py` ではなく `checkpoint.py` に分けた。`generate` が `train.py` を読み込むと「生成に訓練が要る」ように見えるため
- 新しい用語が出る節では用語集 `docs/glossary.md` に追記する
- optimizerは本と同じく `train_model` の外で作って渡す形のままにする。モデルとoptimizerは `.grad` を通して暗黙につながるので、optimizerは必ずそのモデルの `parameters()` で作る。Stage 6でスケジューラやパラメータのグループ分けを入れるときも外で作る

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
