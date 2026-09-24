# Part 1の実行結果

日本語の固定1万文書で学習した記録です。本文の生成例とHTMLの再生には、MPSでの結果を使っています。

| ファイル | 内容 |
|---|---|
| `japanese-10k.json` | MPSで10,000回学習した条件・loss・6入力の生成 |
| `japanese-10k-loss.png` | MPSでのtrain/validation loss curve |
| `japanese-10k-cpu.json` | 同じ設定でCPUを使った結果 |
| `part1.json` | 初期の英語・文字単位の実験記録。現在の教材とは条件が異なる |

## 日本語版の条件

- 実行日：2026-09-24
- 機器：Apple M5 Pro、macOS 26.6
- Python 3.14.7、PyTorch 2.14.0、Matplotlib 3.11.2
- データ：FineWeb-2 Edu Japaneseの固定1万文書。train 9,000文書・validation 1,000文書
- Tokenizer：trainだけから作ったbyte-level BPE、語彙8,192
- モデル：1層・1ヘッド、`d_model=128`、2,320,256パラメータ
- 学習：10,000回、batch size 16、context length 128、learning rate 0.0003、seed 42
- 生成：最大80token、temperature 0.8、seed 100。sampling・greedyの両方を記録

| 指標 | 学習前 | 学習後 |
|---|---:|---:|
| train loss | 9.1897 | 5.5719 |
| validation loss | 9.1951 | 5.7454 |

| 実行環境 | 学習と定期評価の時間 |
|---|---:|
| MPS | 98.5秒 |
| CPU・4スレッド | 305.9秒（約5.1分） |

![日本語データでのloss curve](japanese-10k-loss.png)

時間にはデータ取得・語彙作成・初期化・文章生成・グラフ保存を含みません。lossは各時点で同じ窓を使って測っています。CPUとMPSのlossは小数点以下4桁の表示では一致しましたが、生成文などの完全一致を保証するものではありません。

## 生成の変化

「プログラミングを学ぶには、」を入力すると、学習前は日本語の断片が不規則に並びます。学習後には「ユーザーは増えていきます。」「そのため」のような文らしいつながりが現れました。ただし入力の話題に沿った説明にはなっておらず、greedyでは繰り返しも見られます。

Tokenizerは学習前後で同じです。GPTのパラメータを更新したことで、次tokenの分布と選ばれる続きが変わった、と読み取れます。

データ取得条件は `../dataset-manifest.json`、実行手順は `../README.md` を参照してください。
