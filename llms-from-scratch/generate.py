import torch
import torch.nn as nn

from tokenizer import CharTokenizer


# 本の text_to_token_ids。文字列をトークンIDにencodeし、[1, T] のテンソルにする。
# モデルは [B, T] を受け取るので、promptが1つでも B=1 のバッチの形にそろえる
def text_to_token_ids(text: str, tokenizer: CharTokenizer) -> torch.Tensor:
    encoded = tokenizer.encode(text)
    # unsqueeze(0) は先頭に長さ1の次元を足す。[T] → [1, T]
    return torch.tensor(encoded).unsqueeze(0)


# 本の token_ids_to_text。[1, T] のテンソルをdecodeして文字列に戻す
def token_ids_to_text(token_ids: torch.Tensor, tokenizer: CharTokenizer) -> str:
    # squeeze(0) は先頭の長さ1の次元を外す。[1, T] → [T]。tolist() はテンソルをPythonのリストに戻す
    return tokenizer.decode(token_ids.squeeze(0).tolist())


# 本の generate_text_simple（リスト4-8）に、5.3.1のtemperatureスケーリングを足したもの。
# トークンを1つずつ max_new_tokens 個つなげていく。
#
# 「日本の首都は」から始めると、1回のイテレーションで次のことをする。
#   1. 「日本の首都は」をモデルに入れ、[1, 6, vocab_size] のロジットを得る
#   2. 最後の位置「は」のロジット [1, vocab_size] だけ取り出す。これが「は」の次の文字の予測
#   3. 次の文字を1つ選ぶ
#   4. 入力の末尾に足して「日本の首都は、」にし、次のイテレーションへ
# 1回のイテレーションで生成できるのは1トークンだけなので、長い文章ほど生成に時間がかかる。
# 3の選び方が temperature で変わる。0なら点数が最大の文字（貪欲なデコーディング）、
# 0より大きければ確率に応じたくじ引き（サンプリング）
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        # 位置埋め込みは context_size 個しかないので、入力がそれより長くなったら
        # 末尾 context_size 個だけをモデルに入れる。idx[:, -context_size:] は
        # 「全バッチの、末尾から context_size 個」。それより前の文字はモデルから見えなくなる
        idx_cond = idx[:, -context_size:]
        # 生成では勾配が要らないので、勾配計算用の記録を止める
        with torch.no_grad():
            logits = model(idx_cond)
        # 最後の位置のロジットだけを取り出す。[B, T, vocab_size] → [B, vocab_size]。
        # 他の位置にも「そこまでを見た次の文字の予測」が入っている（「日」→「本」、「日本」→「の」）が、
        # 次に足す1文字を決めるのに要るのは末尾だけ
        logits = logits[:, -1, :]
        # この [B, vocab_size] は、語彙の各文字に付けた点数の表。確率ではなく、負の値もあり、
        # 足しても1にならない。学習後のモデルに「日本の首都は」を入れたときの1行:
        #
        #   ID          0     1  …   412   476  …   480  …    668  …  1981  …
        #   文字       \n  空白  …    、    な  …    の  …     京  …    東  …
        #   logit    2.69  2.58  …  6.82  5.42  …  2.91  …  -1.27  …  0.96  …
        #   softmax後                0.25  0.06     0.005    0.0001   0.0007

        if temperature > 0.0:
            # softmaxではロジットの差がそのまま確率の比になる。temperature で割るとその差が
            # 伸び縮みする。「、」と「な」の差1.4なら
            #   T=1   → 差1.4のまま → 確率は4倍
            #   T=0.5 → 差2.8 → 16倍  小さいほど最大の1つに集中する
            #   T=2   → 差0.7 → 2倍   大きいほど平らになり、低確率の文字も出やすくなる
            probas = torch.softmax(logits / temperature, dim=-1)
            # multinomial は確率分布に従って num_samples 個の添字を引く。[B, vocab_size] → [B, 1]
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            # temperature=0のときは0除算で計算できないのでargmaxで書く
            # argmax は最大値そのものではなく、最大値がある位置（=トークンID）を返す。
            # 上の例なら 6.82 ではなく「、」のID 412。
            # keepdim=True で [B] ではなく [B, 1] にし、次の cat で idx の末尾に付けられる形にする
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        # torch.cat は同じ次元でテンソルをつなぐ。dim=1 は T の方向。[B, T] と [B, 1] → [B, T+1]
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
