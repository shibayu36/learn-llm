# Tiny GPTハンズオンの計画

## 目的と読者

ソフトウェアエンジニアがAIを活用したプロダクトを開発するために、普段使っているLLMの内部で何が起きているかを理解する。最小構成のGPTを自分で実装し、部品の働きと実験結果を自分の言葉で説明できる状態を目指す。

読者はPython・PyTorchの基本を知っているが、Transformer・GPTの内部実装はまだ理解していない。他言語を主に使う人も追えるように、リスト内包表記・ジェネレータ式などを避け、素直なループで書く。

## 資料とサンプルの使い方

- 日本語の本文を読み、読者が自分の作業場所で段階的に実装する。
- `sample/` には参照用の完成コードを置く。読者が直接書き換える作業場所にはしない。
- 本文には手元の実装へ追加・変更する箇所を示し、参照コードとの対応が分かるようにする。
- 本文の正式な形式はMarkdownとする。Part 1で説明の方向性を固めてから、後続のPartを作る。
- 動く図・アニメーション・操作デモが理解を助ける箇所だけ、別ファイルのHTMLを参考資料として用意する。本文全体のHTML版は作らない。
- ユーザーがPart 1を実践した結果を踏まえ、Part 2以降を順に作る。

本文の配置先は `docs/part1.md`、進捗は `docs/handson-progress.md`。参考HTMLの配置先は `docs/demos/` とする。本文から対応するデモを案内する。

## 制作基準

各Partの下書き・実装・推敲・確認に、次の基準を適用する。

### 到達点と説明の順序

章の冒頭で全体像と「何を説明できるようになるか」を示す。モデルの構成図を置き、その章で実装する部分が全体のどこにあるかを分かるようにする。

各主要技術を、次の順で説明する。

1. まず何が問題なのか
2. この技術が解決すること
3. 仕組み。最小限の式・shape・具体例・図を含める
4. 実装
5. 理解を深めるために有効な場合は実験・比較を行う
6. この節で理解したこと

各技術は必要になる理由から説明する。数式は具体例・Tensorのshape・コードと結び付ける。

### 実験と視覚表現の目的

実験は「何を変え、何を観察し、何が分かるか」を先に決める。節ごとの実験は必須にせず、図・数値例・説明で理解できるものはその場で終える。配線やshapeの正しさを確かめる動作確認と、概念を理解するための実験を区別する。

視覚表現は、操作・変化・理解したいことを結び付ける。たとえば「Wvだけを変えると、参照割合は同じまま出力が変わる」という発見につながるデモにする。数値を編集できるだけのデモは置かない。

生成例には、続きを読みたくなる書きかけの文を使う。話者名などのラベルだけで終わらせない。学習前後は同じ入力で比較する。

本文と静的な図はMarkdownに置く。Mermaidなどで表現できる図をHTMLへ移さない。操作が理解を助けるものをHTMLにし、Partごとの1ファイルへまとめて本文からフラグメントでリンクする。HTMLは外部依存なしで開ける構成にし、コードのハイライトにはDark+を使う。

### 下書きと文章レビュー

下書きを作るときも推敲するときも、`core-message-writing` と `japanese-sentence-style` を必ず使う。書く前に想定読者・到達点・伝える核心を定め、読者に重要なことを端的に伝える。理解や実行に必要な説明・手順は残し、一般論や核心をぼかす重複を減らす。

説明用の例と記号を吟味する。入力の「B」とbatch sizeの「B」のような衝突を避け、文字・token・IDなど混同しやすいものを具体例で区別する。未導入の用語を前提に説明しない。

コードの前には「今から何を作るか」、後には「どこに注目すると仕組みが分かるか」を示す。保存先・追加先・置き換える範囲も明記する。

文章レビューは次の3つの観点で行う。

- 読者に不要な情報が含まれていないか
- 記号・用語・説明順など、読んでいて認知しづらい箇所がないか
- 日本語として読みやすいか

制作状況・未確認事項・過去の方針は本文へ混ぜず、計画・進捗資料で管理する。

### コードと実行手順

他言語の経験者が追えるPythonにする。リスト内包表記・ジェネレータ式などを避け、素直なループを使う。コードは責務で分け、重要な計算が構造から読めるようにする。汎用Trainer、テストのためだけのinterface・mock、将来のための過剰な抽象化は作らない。

高レベルなTransformer実装は使わず、Q/K/V・スコア・Mask・softmax・reshape・transpose・ヘッド分割・KV Cache・loss maskingを明示する。Tensor・Linear・Embedding・optimizerなどの基本部品は利用する。

すべてのサンプルコードにPythonの型を付ける。関数の引数・戻り値と設定・コレクションの型を明示し、本文のコード例にも揃えて反映する。Tensorのshapeは型とは別に説明する。

コメントは計算の理解を助けるために使う。Tensorの軸、1tokenずらす理由、未来を隠す理由、勾配計算と更新の違いなどを補う。コードを読み上げるだけのコメントや、作業履歴・会話の経緯は書かない。

`sample/` は参照用の完成コードとし、読者は別の作業場所で実装する。本文を順に進めると完成コードと同じ実装へ到達できるようにし、両者の一致を確認する。前のPartの実装を再利用し、章ごとの差分を追える単位で拡張する。

通常の手順では不要な設定を考えさせない。deviceはautoなど適切な既定値を用意し、設定変更の説明は、その違いを学ぶ実験で必要になったときに出す。

Python・ライブラリは作成時点の最新安定版を使い、uvで環境を管理する。採用バージョンは `.python-version` と `uv.lock` に固定する。

### 比較結果の扱い

条件をそろえて実測し、パラメータ数・loss・生成結果などを示す。「何が変わったか」「考えられる理由」「その結果だけでは断言できないこと」を説明する。lossの低下、文章らしさ、質問や指示へ対応する能力を区別する。

実際のモデルで記録した値と、説明のために置いた数値を区別する。

## Part 1の導入

構成図の直後に、参照コードの未学習GPTで生成する導入を置く。固定した日本語データとBPEの語彙を用意した後、GPTの重みを学習する前に意味の通らない続きを体験する。promptは「プログラミングを学ぶには、」とし、同じpromptをPart末尾でも使って学習前後を比較する。Tokenizerの語彙作成とGPTのパラメータ学習を区別する。

Part 1の操作デモは `docs/demos/part1.html` にまとめる。各デモの学習目的は、この計画の「視覚表現の学習目的」に記録する。

## データセットと実行条件

本文も学習データも日本語とする。Part 1の標準データは `hotchpotch/fineweb-2-edu-japanese` の `small_tokens_cleaned` から固定した1万文書。取得元のrevisionは `180ca004c6a89b590daaad86cb062a07a5353c69`。先頭1万件の既知の重複を除き、空でない異なる本文を1万件選ぶ。seed 42で文書順を並べ替え、train 9,000文書・validation 1,000文書に分ける。JSONLとSHA-256を保存し、以後は同じファイルを使う。

BPEはtrainだけから語彙8,192を作る。1層・1ヘッド・`d_model=128` で2,320,256parameter。10,000回更新した結果、学習前の断片的な出力に、日本語の語句や文末のつながりが現れた。Part 1ではこの変化を体験し、入力の話題を保った文章生成とは区別して観察する。

日本語版は `sample/tiny_gpt/` の標準コードへ統合する。取得は `sample/prepare_data.py`、固定条件は `sample/dataset-manifest.json` に置く。最初のParquetファイル1つだけを取得し、保存済みならハッシュを確認して使う。huggingface-hub 1.32.0・PyArrow 25.0.1・Tokenizers 0.23.2を標準の依存関係にする。

| 選択肢 | 利点 | 難点 | 採否 |
|---|---|---|---|
| FineWeb-2 Edu Japanese＋BPE | 日本語の生成の変化を読み取れる。数百万parameterで実行できる | データ取得と語彙作成が必要 | 主教材 |
| Tiny Shakespeare＋文字単位 | 取得・Tokenizerの実装が短い | 日本語の読み手が生成の変化を実感しにくい | 初期実験の記録として残す |
| 自作の規則的な文章 | 小規模で部品の働きを確認しやすい | 自然言語での能力と混同しやすい | 部品の確認・Instruction Tuning |

データ配布元：FineWeb-2 Edu Japanese: https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese

CPUで1実験5〜10分以内を目標とし、長い実験は任意にする。時間は実測して調整する。GPU必須にせず、実行デバイスの既定値はautoとする。MPSが使える場合はMPS、それ以外はCPUをコード内で選び、本文ではデバイス選択の説明や操作を求めない。環境管理・実行にはuvを使う。

Python・PyTorchなどは作成時点の最新安定版を使う。現在の採用版はPython 3.14.7、PyTorch 2.14.0、Matplotlib 3.11.2。`.python-version` と `uv.lock` に記録して再現できるようにする。

| 条件 | Part 1の標準 |
|---|---:|
| `d_model` | 128 |
| context length | 128token |
| batch size | 16 |
| MLP中間次元 | 512 |
| 層・ヘッド数 | 1層・1ヘッド |
| 語彙数 | 8,192 |
| パラメータ数 | 2,320,256 |
| 更新回数 | 10,000 |
| M5 Proでの学習・評価時間 | CPU約305.9秒 / MPS約98.5秒 |

Positionは学習可能なEmbedding、BlockはPre-LNとする。Part 1はweight tyingなし。各実行でパラメータ数と条件を保存する。

train/validationは窓を切り出す前に分ける。seedを固定し、学習バッチ・評価バッチ・生成の乱数を分ける。比較ではデータ・更新回数・入力token数・生成条件を揃え、処理時間も記録する。seedの反復は発展実験とし、単一seedで得た差を一般則にしない。

## 章・節構成

### 導入

到達目標／uvによる環境構築／参照コードと手元の実装の関係／Tensorのshape／実験条件の記録。

### Part 1：1層・1ヘッドのTiny GPT

1.1 次token予測の全体像

1.2 学習データとTokenization

1.3 Token Embedding

1.4 Position情報

1.5 Q・K・Vと1-head Self-Attention

1.6 Causal Mask

1.7 Feed Forward Network / MLP

1.8 Residual ConnectionとLayerNorm

1.9 Transformer Block

1.10 LM Head・logits・probability・weight tying

1.11 Cross Entropy・Backpropagation・optimizer

1.12 自己回帰生成・greedy・sampling・temperature

1.13 学習前後の生成とtrain/validation loss

到達点：「GPTは何を学習するか」を説明できる。図の候補は入出力対応、Embeddingの参照、QKᵀから重み付き和、Causal Mask、Residual、生成ループ。

### Part 2：Blockを複数層にする

2.1 層を重ねる理由／2.2 独立したBlockのN層化／2.3 1・2・4層の比較／2.4 容量・データ量・計算量・過学習・学習不足／2.5 hidden stateの変化。

比較するもの：parameter数・train/validation loss・学習時間・生成結果。深くすれば必ず改善するとは扱わない。Residual・LayerNormを外す実験は任意で扱う。

### Part 3：Multi-Head Attention

3.1 1ヘッドの制約／3.2 複数のAttention分布／3.3 reshape・transpose・attention・concatのshape／3.4 Output Projection／3.5 1・2・4ヘッドの比較／3.6 ヘッド数とパラメータ数／3.7 headごとのAttention Map。

`d_model` 固定で比較し、ヘッド数だけではQ/K/Vの総パラメータ数が増えないことを確認する。Attention weightを因果的な説明と同一視しない。

### Part 4：KV Cache

4.1 生成時の再計算／4.2 過去のK・Vを再利用できる条件／4.3 層ごとのCacheと追加／4.4 Cacheなし・ありの出力と時間／4.5 PrefillとDecode／4.6 Cacheのメモリ量／4.7 Prompt Cache・Prefix Cacheとの関係。

同じ重み・位置・入力でlogitsを許容誤差内で比較し、greedy生成も確認する。位置のずれやcontext上限での切り捨てを区別する。計時にはwarmup・MPSの同期・複数回測定を含める。短い列では高速化しない場合も記録する。

### Part 5：Instruction Tuning

5.1 Base Modelと質問応答／5.2 学習済み重みの追加更新／5.3 小さなInstruction Dataset／5.4 Base Modelの評価／5.5 SFT／5.6 全token lossとResponse-only loss／5.7 回答形式・追従・汎化・lossの比較／5.8 知識・能力の引き出し方・振る舞いの区別／5.9 System・User・Assistant形式。

簡単な計算・文字列変換・分類・QAなどを用意する。同じBase checkpointから条件を分岐し、小さいlearning rateで追加学習する。学習例と未学習の入力・表現を分け、丸暗記と汎化を区別する。目的が違うloss同士の数値をそのまま優劣として比較しない。

roleもtoken列として扱われるが、商用LLMの振る舞いには他のpost-trainingや推論システムも関わることを補足する。

### 付録

Tokenizerの変更・RoPEへの拡張／プロダクト開発におけるcontext・生成設定・Cache・Fine-tuningの読み替え／理解を確かめる問い。

## この設計を見直す条件

- データや語彙を変える場合：Tokenizerとモデルを組にして作り直す。語彙を増やすだけでは、その言語の能力は得られない。データがメモリへ収まらなくなったら読み込み方式を見直す。
- 長いcontextを扱う場合：学習可能な位置表の上限、Attentionのメモリ量、KV Cacheの位置管理を見直す。
- 実用的なInstruction Modelを目指す場合：モデル規模・事前学習量・Instruction Dataset・評価を見直す。短時間の教材実験の範囲を超える。
- 1つの実装へ多数の実験用分岐が入って読みにくくなる場合：基本のモデルと実験スクリプトの責務を整理する。

## 参考にする文章・コード・HTML

- 文章・コードの提示順：`/Users/shibayu36/development/src/github.com/shibayu36/book-realtime-communication-server`
- サンプルの責務分割・段階的拡張：`/Users/shibayu36/development/src/github.com/shibayu36/sample-realtime-communication-server/CLAUDE.md` と実装
- HTMLの雰囲気：`/Users/shibayu36/obsidian/エンジニア知識メモ/20260921 LLMを実装しながら理解するハンズオン.html`

参考HTMLは淡い背景と大きな図・操作デモを基本とする。コードを載せる場合のハイライトはDark+にする。Markdownのコード色は閲覧環境に依存する。実測結果と、説明用の模式図・手で設定した数値を区別する。


## 視覚表現の学習目的

各デモは「操作すると何が変わり、何が分かるか」を先に決める。数値を編集できるだけのデモは置かない。

| 本文 | デモ | 理解させたいこと |
|---|---|---|
| 1.5 | 内積→softmax | 向き・長さによるscoreの違いが、合計1の参照割合になる |
| 1.5 | Valueを順に足す | 情報を集める処理の実体は、Valueの重み付き和である |
| 1.5 | Wq・Wk・Wvの編集 | 入力から計算するQ/K/Vと学習するWは違う。参照割合と集める情報も別である |
| 1.6 | 未来の正解を隠す | 学習時に入力の先にある正解を参照すると、生成時に使えない経路になる |
| 1.12 | greedy・sampling・temperature | モデルの重みを変えずに、次tokenの選び方を変えられる |
| 導入・1.12・1.13 | 日本語の生成を1tokenずつ再生 | 選んだtokenを次の入力へ戻す。学習前後で分布と続きが変わる |

Self-Attentionの参考は `/Users/shibayu36/obsidian/エンジニア知識メモ/20260923 Self Attentionの仕組み（内積・softmax・重み付き和・QKV・Causal・Multi-Head）を視覚的に理解する.html`。内積の矢印、softmaxへの変換、矢印をつなぐ重み付き和、Wの編集という見せ方をPart 1へ取り入れる。計算は本文のscaled dot-product attentionにそろえ、説明用の数値を学習結果とは扱わない。Multi-HeadはPart 3の参考にする。
