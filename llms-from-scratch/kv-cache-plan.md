# KVキャッシュの計画

## 目的と方針

生成で1文字足すたびにprompt全体を計算し直している `generate` に、KVキャッシュを足す。各Attentionヘッドが作った過去のキーと値を次の回に持ち越し、新しい1文字の分だけ計算する。

本の本文にはKVキャッシュの節がない。著者のGitHubの補足資料 `ch04/03_kv-cache` に相当する。

目的は2つ（2026-09-29に決定）。

1. Causal Attentionの「過去の位置の出力は、後ろに文字が増えても変わらない」という性質を、実装と一致確認で確かめる
2. 「プロンプトキャッシュは何をキャッシュしているのか」「なぜeffortを変えるとキャッシュが壊れるのか」「なぜキャッシュに有効期限があるのか」に、キャッシュの中身と大きさの実測から答える

解説HTMLは [docs/demos/KVキャッシュで生成が速くなる理由.html](../docs/demos/KVキャッシュで生成が速くなる理由.html)。

決めたこと。

- キャッシュはモデルの中に持たず、`generate` が作って `forward` に渡す。キャッシュは「そこまでに入れた文字列」ごとに決まるものでモデルの一部ではないので、モデルと切り離す。前方の文字列が同じなら別の会話でも使い回せる。モデルは「入力だけで出力が決まる」ままにする
- 対応するモデルは `JpCharGPT2` だけ（2026-09-29に `GPT` も対象から外した）。`GPT`・`OneLayerOneHeadGPT`・`OneHeadGPT` は凍結した理解用の `CausalAttention`・`MultiHeadAttention`・`TransformerBlock` を使い、`cache` を受け取らない
- `generate` は変えず、`generate_with_kv_cache` を別関数にする（2026-09-29に「1つの関数で `use_kv_cache` を場合分けする」から変更。部品クラスと同じく、理解用とキャッシュ版を分けたほうが読みやすい）。「次の文字を1つ選ぶ」は `select_next_token` に切り出して両方から呼ぶ。`main.py generate` は `--kv-cache` を付けたときだけ `generate_with_kv_cache` を使う。`evaluate` は `generate` のままにして、記録済みの結果と比べられる状態を保つ
- promptの文字数と `max_new_tokens` の合計が `context_length` を超えるときは、キャッシュありでは生成を始める前に例外で止める。末尾を切り詰めて位置を振り直す動きをキャッシュで再現すると毎回作り直しになり、その経路は実際の使い方（prompt数十文字 + 100文字、上限256か1,024）で通らないため。キャッシュなしは本どおり末尾を切り詰める
- キャッシュのクラスは `kv_cache.py` に分ける。`gpt.py` はモデルの部品だけにする
- streamingはしない。`main.py generate` は生成時間と1文字あたりの時間だけ出す
- 速度とキャッシュの大きさは `kv-cache-results.md` に記録する。モデルの質の結果ではないので `evaluation-results.md` には入れない

## キャッシュの形

`kv_cache.py` に2つのクラスを置く。

| クラス | 持つもの | 読む側 |
|---|---|---|
| `HeadKVCache` | 1つのヘッドの `keys`・`values`（各 `[B, T_past, head_dim]`）。`append(keys, values)` で今回の分を後ろにつなぎ、全体を返す | `CausalAttentionWithKVCache` |
| `KVCache` | `blocks[層][ヘッド]` に `HeadKVCache`。`KVCache(model)` で `model.trf_blocks` を歩いて空の入れ物を作る。`num_tokens()` で溜まった文字数を返す | `JpCharGPT2` の `forward` |

`kv_cache.py` は `gpt.py` を import しない（`gpt.py` が `HeadKVCache` を import するため）。対応クラスの判定は `generate_with_kv_cache` で行う。

## 各部品の変更

| 部品 | 変更 |
|---|---|
| `CausalAttentionWithKVCache.forward(x, cache)` | `CausalAttention` を継承して `forward` だけ書く（重みとmaskは同じなので `state_dict` のキー名は変わらない）。`start = cache.num_tokens()` とし、`keys, values = cache.append(keys, values)` で過去とつなぐ。Causal maskは `mask[start:start+T_new, :start+T_new]` と「新しい文字の行」だけを切り出す |
| `MultiHeadAttentionWithKVCache.forward(x, cache)` | `MultiHeadAttention` と同じ構造で、ヘッドを `CausalAttentionWithKVCache` で作る（ヘッドのクラスは `__init__` で決まるので継承しない）。`cache` は `list[HeadKVCache]` で、ヘッドごとに `cache[i]` を渡す |
| `TransformerBlockWithKVCache.forward(x, cache)` | `TransformerBlock` を継承して `forward` だけ書き、`self.att(x, cache)` と渡す |
| `JpCharGPT2.forward(in_idx, cache=None)` | 部品を WithKVCache の3クラスにする。`cache` が `None` なら入口で使い捨ての `KVCache(self)` を作る。その後は無条件に `start = cache.num_tokens()` とし、位置埋め込みを `arange(start, start + T_new)` にする。`nn.Sequential` の一括呼び出しはforループにして `cache.blocks[layer]` を渡す。`state_dict` のキー名は変わらない |
| `generate_with_kv_cache(model, ...)` | `JpCharGPT2` 以外は `TypeError`。上限の確認と `KVCache(model)` の作成。モデルに入れる `idx_input` は1回目がprompt全体、2回目からは足したばかりの `idx_next`。`cache` の中身は読まない（`forward` が `cache` に書き込む副作用に `generate` の制御を依存させないため。2026-09-30に `idx[:, cache.num_tokens():]` を使う形から変更）。`generate` は `idx[:, -context_size:]`（全文）のまま |
| `main.py generate` | `--kv-cache` と、生成時間・1文字あたりの時間の表示 |

理解用の `CausalAttention`・`MultiHeadAttention`・`TransformerBlock` は凍結し、`forward(x)` のまま変えない。旧クラスと `GPT` はこちらを使う。WithKVCache 側は `cache` を必須にし、`None` の場合分けは `JpCharGPT2` の入口の1か所だけにする（部品ごとに「`None` なら `None` のまま渡す」を書く形は分岐が4段に散って読みにくいので、2026-09-29にこの形に決めた）。キャッシュなしの呼び出しでは使い捨ての空キャッシュに全文字を一度に入れるので `start=0` で、計算は今までと同じ。

Causal maskの切り出しは、キャッシュなし（`start=0`）なら今までと同じ左上の正方形、1文字だけ入れたときはその文字の1行（過去全部が0で見てよい）になる。

## 作業項目

各段階でcommitする。確認用スクリプトは `tmp/` に置きcommitしない。

1. `main.py generate` に生成時間と1文字あたりの時間を出す
2. `kv_cache.py`（`HeadKVCache`・`KVCache`）と `CausalAttentionWithKVCache.forward(x, cache)`。確認: 1ヘッド単体で「日本の首都は」を一括で通した出力と1文字ずつ通した出力の最大差。理解用クラスが変わっていないことは `evaluate --run-name l4h4-d128-s5000-lr1e-3` の全バッチloss 3.094512 の一致で見る
3. `MultiHeadAttentionWithKVCache`・`TransformerBlockWithKVCache`・`JpCharGPT2` の配線。確認: `ku-nlp-gpt2-small-char` で、prompt全体を一括で通した末尾logitsと1文字ずつキャッシュで送った末尾logitsの最大差。`cache=None` の経路は `evaluate --run-name ku-nlp-gpt2-small-char` の全バッチlossの一致で見る
4. `generate_with_kv_cache` と `main.py generate --kv-cache`。確認: 固定10promptのgreedy 100文字が `generate` と完全一致。同seedのsamplingも一致するか。上限超えで例外になること。`visualize-attention` が変わらないこと（hookは `inputs[0]` しか読まない）
5. `kv-cache-results.md`: ku-nlpで100・200・300文字の生成時間（CPU）をキャッシュあり・なしで比べる。ku-nlpのキャッシュの大きさ（層数 × ヘッド数 × 2 × 文字数 × head_dim × 4バイト）を出し、3つの問いへの考察を書く
6. `progress.md`・`docs/glossary.md`（KVキャッシュ）・解説HTMLの「今の実装：キャッシュなし」の節を実装後の状態に合わせる

## この設計が崩れる将来の変更

- `MultiHeadAttention` を分割方式（1つの `W_query` を `view` で切る形）に変えると、キーと値はヘッドごとでなく `[B, H, T, head_dim]` の1テンソルになり、`blocks[層][ヘッド]` の入れ子が崩れる。そのときは `KVCache` を層ごと1テンソルにする
- `context_length` を超えて生成し続けたいなら、学習済みの絶対位置埋め込みでは古いキーと値の先頭を削っても位置がずれて結果が変わる。相対位置（RoPEなど）に変えない限り作り直ししかない
- 長さの違う複数promptをバッチで生成するなら、Causal maskのほかにpaddingのmaskが要る。`append` は全バッチ同じ長さを仮定している
- fine-tuning（第7章）は訓練経路（`cache=None`）だけを使うので影響しない
