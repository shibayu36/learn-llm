import torch
import torch.nn as nn

from gpt import JpCharGPT2
from kv_cache import KVCache
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


# 末尾の位置のロジット [B, vocab_size] から、次の文字を1つ選んで [B, 1] で返す。
# 選び方は temperature で変わる。0なら点数が最大の文字（貪欲なデコーディング）、
# 0より大きければ確率に応じたくじ引き（サンプリング）
def select_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
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
        return torch.multinomial(probas, num_samples=1)
    # temperature=0のときは0除算で計算できないのでargmaxで書く
    # argmax は最大値そのものではなく、最大値がある位置（=トークンID）を返す。
    # 上の例なら 6.82 ではなく「、」のID 412。
    # keepdim=True で [B] ではなく [B, 1] にし、呼び出し側の cat で idx の末尾に付けられる形にする
    return torch.argmax(logits, dim=-1, keepdim=True)


# 本の generate_text_simple（リスト4-8）に、5.3.1のtemperatureスケーリングを足したもの。
# トークンを1つずつ max_new_tokens 個つなげていく。
#
# 「日本の首都は」から始めると、1回のイテレーションで次のことをする。
#   1. 「日本の首都は」をモデルに入れ、[1, 6, vocab_size] のロジットを得る
#   2. 最後の位置「は」のロジット [1, vocab_size] だけ取り出す。これが「は」の次の文字の予測
#   3. 次の文字を1つ選ぶ（select_next_token）
#   4. 入力の末尾に足して「日本の首都は、」にし、次のイテレーションへ
# 1回のイテレーションで生成できるのは1トークンだけなので、長い文章ほど生成に時間がかかる
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
        idx_next = select_next_token(logits[:, -1, :], temperature)
        # torch.cat は同じ次元でテンソルをつなぐ。dim=1 は T の方向。[B, T] と [B, 1] → [B, T+1]
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# generate にKVキャッシュを足したもの。毎回全文をモデルに入れる代わりに、まだ入れていない
# トークンだけを入れる。1回目はprompt全体、2回目からは足したばかりの1トークン。過去の
# トークンのK・Vは KVCache に持ち越すので、結果は generate と同じまま計算量が減る
def generate_with_kv_cache(
    model: JpCharGPT2,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
) -> torch.Tensor:
    if not isinstance(model, JpCharGPT2):
        raise TypeError(f"{type(model).__name__} はKVキャッシュに対応していない")
    # generate のように末尾を切り詰めると全トークンの位置がずれ、キャッシュを毎回作り直すことに
    # なる。シンプルさを優先して、位置埋め込みの数を超える生成は始める前に止める
    if idx.shape[1] + max_new_tokens > context_size:
        raise ValueError(
            f"prompt {idx.shape[1]}トークン + 生成 {max_new_tokens}トークンが"
            f" context_size {context_size} を超える"
        )
    cache = KVCache(model)
    # 1回目はprompt全体を入れる
    idx_input = idx
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(idx_input, cache=cache)
        idx_next = select_next_token(logits[:, -1, :], temperature)
        idx = torch.cat((idx, idx_next), dim=1)
        # 2回目からは、足したばかりの1トークンだけを入れる
        idx_input = idx_next
    return idx
