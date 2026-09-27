# llms-from-scratch の進捗

計画は `plan.md`。ここでは各Stageの作業状態と、実行して分かったことを記録する。チェックは実行して確認できたときに付ける。

## 現在の作業

Stage 4（複数層）を進行中（2026-09-27）。`OneHeadGPT`（`n_layers` 可変・1ヘッド）を実装し、4層の学習・評価と、参考実験の1層20,000stepまで完了。残りは2層の学習、4層のhidden stateの観察、記録の更新。`config.py` は `n_layers=1`・`steps=5000` に戻してある。

Stage 4・5の `d_model` は128に決めた。1層1ヘッドの基準runは `l1h1-d128-s5000-lr1e-3`。5,000step・学習率1e-3・batch 16は維持する。層数の切り替えはCLI引数ではなく `config.py` を手で変える。学習時間に上限は設けず、遅くなったらMPSを検討する。固定10promptの学習前出力は未記録。

- Stage 4の `OneHeadGPT` の実装と4層・20,000stepの結果、`evaluation-results.md` の構成変更は未commit。実装と結果で2つに分けてcommitする
- `evaluation-results.md` は「変えた条件のカテゴリ（層数・幅・更新回数と学習率）を `##`、問いを `###`」の構成にした。新しいカテゴリ・問いは上に足す。各runの学習曲線と全20出力は末尾にまとめず、そのrunを最初に使った問いの節の末尾に置く。計画外の参考実験は1段落に収める
- `runs/` はディレクトリごとcommitしない。model.pt を入れないなら config.json や metrics.json だけ残しても再現できないため。残したい数値や生成結果は `evaluation-results.md` に書く
- `load_run` は復元だけを行い、`model.eval()` は生成・評価する側で呼ぶ。読み込んだモデルを何に使うかは呼び出し側が決めることなので
- `generate.py` のロジットの説明は、学習後のモデルに「日本の首都は」を入れて観察した実値の表にした。観察に使ったスクリプトは `tmp/show_logits.py`（commitしない）
- 保存・読み込みは `train.py` ではなく `checkpoint.py` に分けた。`generate` が `train.py` を読み込むと「生成に訓練が要る」ように見えるため
- 用語集 `docs/glossary.md` を作った。新しい用語が出る節では追記する
- optimizerは本と同じく `train_model` の外で作って渡す形のままにする。モデルとoptimizerは `.grad` を通して暗黙につながるので、optimizerは必ずそのモデルの `parameters()` で作る。Stage 6でスケジューラやパラメータのグループ分けを入れるときも外で作る

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
- [ ] 2層を学習・評価し、1・4層と並べる
- [ ] 4層のrunで層ごとのhidden stateを観察
- [ ] 結果と考察を記録

## Stage 5：Multi-head Attention

- [ ] `MultiHeadAttentionWrapper`
- [ ] `MultiHeadAttention`（分割方式）と `Config.n_heads`、`GPTModel`
- [ ] `n_heads=1` で `OneHeadGPT` と結果が変わらないことを確認
- [ ] 1・2・4ヘッドの比較（`config.py` を手で変更）
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

### Stage 2：生成と損失（2026-09-25）

- 学習前の「日本の首都は」の続き（30文字）。本の "Featureiman Byeswickattribute argue" と同じく、訓練前はでたらめな文字列になる
  - T=0（greedy）：`盾!ワ銛剖般篠諱券倖貧†替凧酩語盛府柑肛ゲ敬↔登存Y厖岬翅巷`。2回呼んで同じ
  - T=0.8：`塗葯樫拠瘍校経ρカ啄ぎ七ơ蔓襠祐潔腰椒褪么混薙郡沈鯨鹸櫨滴様`、`ド胎数稔苔旧ぐ此銭句豪遅設拮瘡訂戯旗己馴劫怯聲憤渤髙爪噂⣦嫡`。呼ぶたびに違う
- 設計の決定：本の `generate_text_simple` は作らず、最初から `generate(temperature)` にした。greedyは T→0 の極限なので `temperature=0` の分岐で表す（0で割るとsoftmaxが壊れるため）。top-kは評価に使わず、学習後のT=0.8の出力に尻尾の文字が混ざって読めないときだけ足す
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

- 2,000stepでも下がり続けている（最後の200stepで0.05）。パープレキシティは `exp(4.26)=71`。学習時間に余裕があるので、段階3で `steps` を増やす（5,000なら約2分）
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
- 決定：この条件（5,000step・学習率1e-3）を以降の基準にする。`config.py` を書き換え、plan.md の基準値も更新した

### Stage 3：d_model 64 → 128の比較（2026-09-26）

`config.py` の `d_model` だけを128に変え、`l1h1-d128-s5000-lr1e-3` として初期状態から学習した。比較元は `l1h1-s5000-lr1e-3`。batch 16・5,000step・学習率0.001など他の設定は同じ。batch sizeの変更実験は行っていない。

- 全バッチvalidation lossは3.717512 → 3.376540（約9.2%低下）。評価対象は両方とも657,408token
- パラメータ数は580,800 → 1,251,712（約2.16倍）。学習時間は111.8秒 → 163.2秒（約1.46倍、定期評価・生成を含む）
- samplingでは「確認していくことが必要です」などの語句が見られるが、プログラミングの学び方を説明できていない。greedyは全10promptが「そのため」の反復に陥り、lossの改善と生成文の質は一致しなかった
- 手動確認：全20出力を同じ条件の64次元モデルと見比べ、学習曲線のPNGを開いて軸・ラベル・lossの推移を確認した
- 自動確認：設定差分がd_modelだけであること、保存した語彙・評価token数・生成条件の一致、26時点の学習履歴が有限値であることを確認した。モデルを再読み込みして全20出力を再生成し、保存済み結果との完全一致も確認した

この結果を踏まえ、Stage 4・5の幅は128に決めた（2026-09-27）。詳細と全20出力は `evaluation-results.md` に残した。

### Stage 3：Token embeddingの地図（2026-09-26）

`l1h1-s5000-lr1e-3` の `tok_emb.weight` をPCAで2次元にし、自己完結HTML（検索・近傍一覧・頻度フィルタ付き）で観察した。数値と観察は `visualization-results.md`。

- PCAは自前実装（平均を引いて `torch.linalg.svd`）。2軸の寄与率は3.6%・3.3%で、64次元をほぼ均等に使っている（均等なら1軸1.6%）。d_model 128のrunで2.5%・2.1%、2,000stepのrunで3.2%・3.2%と、幅や学習量を変えても同じ
- PCAの図の近さは元の空間の近さを表さない。「高」と図で重なる態・制・心の類似度は−0.02〜0.13、類似度0.42の低は図で離れる。出現100回以上の全組で、2次元の距離と類似度の相関係数は−0.13
- 近傍候補を頻度上位300に限ると短・低・西が外れるので、HTML側は「最小出現回数」のスライダー（既定100）で絞る形にした
- 出現回数別の平均ノルムは1〜10回で4.84、11〜100回で4.95、101〜1,000回で5.24、1,001回以上で5.57。稀な文字の4.84は、初期値のノルム8にweight decayの倍率 (1−1e-4)^5000 ≈ 0.61 を掛けた値と一致し、ほぼ学習されていない。近傍候補を出現100回以上に絞る根拠
- 意味の近さ（長↔短、高↔低、東↔西、北・南）は元の空間の近傍に、表記の種類（ひらがな・漢字・カタカナ・数字）は2次元の配置に現れた
- HTMLは `templates/embedding_map.html` をテンプレートとしてcommitし、Pythonがデータを埋め込んで `runs/<run>/embedding_map.html` に書く。検証はヘッドレスChromeのスクショ・console・iframe操作テストで行った
- PCAの図では近傍が隣に来ないので、umap-learn 0.5.12を足してUMAP（metric cosine）に切り替えられるようにした。設定は「コサイン上位10の文字が図の最近傍10に入る個数」の全文字平均で選び、n_neighbors 5・min_dist 0にした（既定の15・0.1で1.4個、採用値で2.2個。偶然なら0.06個、PCAの図では0.19個）
- UMAPは全体の一致数がseedで安定する一方、特定の組が図で何番目に近いかはseedで大きく変わる（高→低が603位・5位・5位）。図の隣接だけで「近い」と言わず、近傍一覧の類似度で確かめる

### Stage 3：Hidden stateの変化（2026-09-26）

`l1h1-s5000-lr1e-3` で、前文の違う6文の「行」（位置9）と「の」（位置10）について、Layer 0とLayer 1のhidden stateを `register_forward_hook` で取り出し、層ごとの「次の文字の予測（logit lens）」と「validation先頭4万位置の中で近い文脈」の表をHTMLにした。数値と観察は `visualization-results.md`。

- 最初はPCAで各層の点を結ぶ軌跡図にしたが、「ブロックを通ると文脈で変わる」以上のことが読めずやめた。方針変更の理由は `visualization-plan.md`
- hidden stateを `final_norm` → `out_head` に通すと途中の層でも次の文字の予測として読める。近い文脈は、同じ層のvalidationのベクトルとのコサイン類似度で、embedding地図の近傍一覧の文脈版にあたる
- 「行」はLayer 1で、銀行・旅行のあとは動詞の行（行う・行った）の仲間、実行・発行のあとは熟語の行（銀行・非行・先行）の仲間になった。前文と直前の単語を入れ替えた文を足すと両方が効いており、直前の1文字の方が動かす量は大きかった
- 「の」は6文とも予測が平坦（最大0.02）で近傍も名詞に続く「の」ばかりになり、文脈でほとんど変わらない
- 予測上位に前文の末尾の文字（楽しもう→う、受領した→し）が入る。ヘッドが前の文字を写している可能性があり、Attentionの重みを見るときに確かめる
- 特殊トークンは `<|endoftext|>` を「␃」、改行を「␤」、`<|unk|>` を「�」の1文字にして、前後の文字を切り出す添字がずれないようにした

### Stage 4：OneHeadGPT と1層の一致確認（2026-09-27）

`Config.n_layers`（既定1）を足し、`TransformerBlock` を `nn.Sequential` に `n_layers` 個並べた `OneHeadGPT` を `gpt.py` に追加した。`main.py train` は `OneHeadGPT` で学習し、`checkpoint.load_run` はクラス名で `OneLayerOneHeadGPT` と `OneHeadGPT` を組み立て分ける。既存3runのローカルの `config.json` には `"n_layers": 1` を手で足した。クラス名は最初 `SingleHeadGPT` にしたが、`OneLayerOneHeadGPT` と語をそろえて `OneHeadGPT` に改名した。Stage 5のMulti-head版は `GPTModel` にする。

`tmp/check_gpt_model.py`（commitしない）で `n_layers=1` の `OneHeadGPT` が `OneLayerOneHeadGPT` と同じものであることを確かめた。

- 同じseed（42）で作ると、パラメータ数1,251,712・17個のパラメータの初期値がすべて一致。パラメータの作られる順番が同じなので乱数の消費も同じになる。`state_dict` のキーは `trf_block.` と `trf_blocks.0.` で違うだけ
- ランダムな入力 `[2, 16]` のlogitsが完全一致（`torch.equal`）。cross entropyで `backward` した勾配も全パラメータで完全一致。forwardとbackwardが同じなら学習の経過も同じになるので、1層は学習し直さない
- `l1h1-d128-s5000-lr1e-3` の重みをキー名を付け替えて `OneHeadGPT` に流し込み、全バッチvalidation lossを測ると 3.3765399842247414 で、保存済みの値と小数点以下すべて一致。評価token数も657,408で同じ
- 既存runの `generate` も変わらず動く（「日本の首都は、大阪の中には、大人の中でも、大阪では、」）

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

参考実験。「4層の差はstep不足では」を確かめるため、1層・128次元を20,000stepで学習した（`l1h1-d128-s20000-lr1e-3`、677.2秒）。全バッチvalidation lossは3.113614で4層5,000stepの3.067810に届かず、trainを約14周しても過剰適合せず、greedyの反復も変わらなかった。事前の「等比で縮むなら3.20止まり」の予想は外れ、減り幅は等比より遅く縮む。学習中に `SingleHeadGPT` を `OneHeadGPT` に改名したため、このrunの `config.json` の `model` は手で直した。
