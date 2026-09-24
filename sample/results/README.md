# Part 1の実行結果

日本語の固定1万文書で学習した記録です。本文の生成例とHTMLの再生にはこの結果を使っています。

| ファイル | 内容 |
|---|---|
| `japanese-10k.json` | 10,000回学習した条件・loss・6入力の生成 |
| `japanese-10k-loss.png` | train/validation loss curve |
| `part1.json` | 初期の英語・文字単位の実験記録。現在の教材とは条件が異なる |

## 日本語版の条件

- 実行日：2026-09-24
- 機器：Apple M5 Pro、macOS 26.6
- Python 3.14.7、PyTorch 2.14.0、Matplotlib 3.11.2
- データ：FineWeb-2 Edu Japaneseの固定1万文書。train 9,000文書・validation 1,000文書
- Tokenizer：1文字1token。trainに出てきた文字にEOS・UNKを加えた語彙4,052
- モデル：1層・1ヘッド、`d_model=128`、1,256,276パラメータ
- 学習：10,000回、batch size 16、context length 128、learning rate 0.0003、seed 42
- 生成：最大80token、temperature 0.8、seed 100。sampling・greedyの両方を記録

| 指標 | 学習前 | 学習後 |
|---|---:|---:|
| train loss | 8.5125 | 3.5022 |
| validation loss | 8.5111 | 3.5536 |

学習と定期評価の時間は約69秒でした。データ取得・語彙作成・初期化・文章生成・グラフ保存を含みません。lossは各時点で同じ窓を使って測っています。

![日本語データでのloss curve](japanese-10k-loss.png)

## 生成の変化

「プログラミングを学ぶには、」を入力すると、学習前は語彙の文字がでたらめに並びます。学習後には「次世代に入り」「ことができます」「状況ではあります」のような語句や文末のつながりが現れました。ただし入力の話題に沿った説明にはなっておらず、greedyでは「そのために、」の繰り返しが続きます。

Tokenizerは学習前後で同じです。GPTのパラメータを更新したことで、次tokenの分布と選ばれる続きが変わった、と読み取れます。

データ取得条件は `../dataset-manifest.json`、実行手順は `../README.md` を参照してください。
