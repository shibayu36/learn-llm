# 用語集

本文（`docs/part1.md` 以降）で使う用語と表記を定める。同じ概念は同じ語で呼び、初出で必要な併記をしたあとは統一した表記だけを使う。

## 基本方針

- Transformer・GPTの構成要素と計算の名前は英語表記にし、カタカナ訳を使わない。読者が論文・ライブラリ・他の解説を読むときに同じ語で探せるようにするため
- 処理や状態を表す一般的な動作は日本語にする（学習、生成、更新、参照）
- Pythonの識別子（`batch_size`、`scores`、`vocab_size` など）はコードのまま書き、本文の呼び方と対応付けを1度示す
- 1つの語に2つの意味を持たせない。特に「重み」「スコア」「層」は下の定義に従う

## 用語一覧

### モデルと構成要素

| 統一表記 | 使わない表記 | 補足 |
|---|---|---|
| GPT、モデル、Tiny GPT | ネットワーク | クラス名は `TinyGPT` |
| Transformer Block、Block | ブロック | |
| Attention、Self-Attention | アテンション、注意機構 | |
| Causal Self-Attention | Causal Attention | 1.6で定義 |
| Causal Mask、Mask | マスク | |
| Q・K・V、Wq・Wk・Wv | Q/K/V、Wq/Wk/Wv | 区切りは「・」 |
| MLP | | 1.7の見出しで Feed Forward Network と併記。クラス名は `FeedForward` |
| Residual Connection | 残差、残差接続 | |
| LayerNorm | 層正規化 | 「正規化」は動詞としてのみ使う |
| Embedding、Token Embedding | 埋め込み、エンベディング | |
| Position情報 | 位置情報 | 見出しに合わせる |
| LM Head | | |
| Tokenizer、token | トークナイザ、トークン | Tokenization は節見出しのみ |
| 1-head、1ヘッド | | 見出しは「1-head」、本文は「1ヘッド」 |
| 層 | | Blockを1回通すことを1層と数える。Linear などの部品を「層」と呼ばない。部品は「線形変換」「部品」「モジュール」 |

### 値と計算

| 統一表記 | 使わない表記 | 補足 |
|---|---|---|
| パラメータ | 重み（学習対象全体の意味で） | `optimizer.step` で更新される値の総称。出力ラベルは「パラメータ数:」 |
| 重み | | 特定の重み行列（Embeddingの重み、Wq など）を指すときだけ使う |
| Attention weight | 重み、weight（単独） | softmax後の参照割合。説明の言い換えとして「参照する割合」は使ってよい |
| スコア | score（本文中） | Attentionの QKᵀ の値だけを指す。変数名は `scores`。初出で「スコア（score）」と併記 |
| logit、logits | スコア（1.10以降）、ロジット | 次token候補ごとの値。1.10で定義したあとは「logit」で通す |
| loss | Loss、損失 | 固有名「Cross Entropy Loss」だけ大文字 |
| 勾配、勾配計算 | 勾配の計算 | |
| 学習 | 訓練、トレーニング | Tokenizerの語彙作成と区別する |
| 更新、10,000回の更新 | 10,000回の学習 | 更新回数を数えるときの単位 |
| 生成 | 推論 | 次tokenを選んで入力へ戻す処理全体 |
| greedy、sampling、temperature | サンプリング、温度 | 表の初出のみ「greedy decoding」 |
| learning rate | 学習率 | 本文の初出で `learning_rate` と対応付ける |
| Backpropagation、backward | 誤差逆伝播 | 1.11で対応を説明 |
| optimizer、AdamW | 最適化器 | |
| Fine-tuning | | 初出で「学習済みパラメータの追加学習」と併記 |
| hidden state | | 単独では使わない。「Blockの出力（hidden state）」の形で併記 |

### データと入出力

| 統一表記 | 使わない表記 | 補足 |
|---|---|---|
| train、validation | 学習データ・未使用データ（比較文脈で） | 比較や loss の文脈では train / validation を軸に書く。「1.2 学習データとTokenization」の見出しは例外 |
| 系列 | シーケンス | 1つの入力に含まれるtokenの並び。「token列」「ID列」は token と ID を区別する目的で使ってよい |
| 窓 | ウィンドウ | 学習データから切り出す固定長の系列 |
| バッチ、バッチサイズ | batch、batch size | 識別子は `batch_size`、shapeは `B` |
| prompt | 入力文（生成の起点の意味で） | 生成の起点として渡す入力。1.12で定義。「入力文」は学習データの文にだけ使う |
| 語彙、語彙数 | vocab | 識別子は `vocab_size`。Value の `V` と区別する |
| EOS、終端token | | 初出で「専用の終端token（EOS）」と併記 |
| UNK | 未知語token | 初出で「語彙にない文字を表すtoken（UNK）」と併記 |
| seed | シード | |
| Tensor、shape | テンソル | |

### 記号

| 統一表記 | 使わない表記 | 補足 |
|---|---|---|
| `B`、`T`、`D` | | `B` はバッチサイズ、`T` は1系列のtoken数、`D` は `d_model` |
| `context_length` | 文脈長 | `T` の上限。`T` は常に最大値とは限らない |
| `d_model`、`d_k` | | |
| `[B, T]` | | Mermaid図内だけ構文回避のため全角「［B, T］」 |
| −∞ | -∞ | U+2212 のマイナス記号 |
| B×T | B*T | 本文・コメントとも「×」 |
| 〜 | ~ | 範囲は全角の「〜」 |

### 資料の構成要素

| 統一表記 | 使わない表記 | 補足 |
|---|---|---|
| Part 1、1.2 | パート1、第1部、1章 | 節参照は「1.2で」の形 |
| 操作デモ： | 参考：、生成の観察： | `docs/demos/` へのリンク行の前置き |
| 参照コード、完成コード | サンプルコード | `sample/` のコード。「参照用の完成コード」と初出で対応付ける |
| 作業用ディレクトリ | 作業場所 | 読者が実装する場所 |
| 確認コード、動作確認 | | 配線と shape を確かめるコード。概念を理解する「実験」と区別する |

## 日本語の表記

- 「揃える」「分かる」「できる」「行う」は漢字・かなをこのとおりに固定する
- 数は算用数字にする（1つずつ、4つのprompt）
- 「次token」は複合語、「次のtoken」は地の文で使い分けてよい
