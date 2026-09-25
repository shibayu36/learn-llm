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
#   3. 次の文字を1つ選ぶ（学習後なら「東」など）
#   4. 入力の末尾に足して「日本の首都は東」にし、次のイテレーションへ
# 1回のイテレーションで生成できるのは1トークンだけなので、長い文章ほど生成に時間がかかる
#
# 3で選ぶ方法が temperature で変わる。
#   temperature=0: 最も確率が高い文字を選ぶ（貪欲なデコーディング）。毎回同じ文になる
#   temperature>0: ロジットを temperature で割ってからsoftmaxで確率にし、確率に応じてくじ引きで選ぶ。
#     ロジット [2.0, 1.0, -1.0] なら
#       T=1   → softmax [0.71, 0.26, 0.04]
#       T=0.5 → [4.0, 2.0, -2.0] → [0.88, 0.12, 0.00]  小さいほど最大の1つに集中する
#       T=2   → [1.0, 0.5, -0.5] → [0.51, 0.31, 0.19]  大きいほど平らになり、低確率の文字も出やすくなる
#     T→0 の極限が「最大の1つだけ確率1」で、これがtemperature=0のargmaxと同じ。
#     ただし0で割ると inf になってsoftmaxが壊れるので、0のときだけargmaxで書く
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
        # torch.no_grad() の中では、訓練で使う勾配の計算に必要な記録を取らない。
        # 生成では不要なので、メモリと時間の節約になる
        with torch.no_grad():
            logits = model(idx_cond)
        # 最後の位置のロジットだけを取り出す。[B, T, vocab_size] → [B, vocab_size]
        logits = logits[:, -1, :]
        if temperature > 0.0:
            probas = torch.softmax(logits / temperature, dim=-1)
            # multinomial は確率分布に従って num_samples 個の添字を引く。[B, vocab_size] → [B, 1]
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            # argmax は最大値そのものではなく、最大値がある位置（=トークンID）を返す。
            # keepdim=True で [B] ではなく [B, 1] にし、次の cat で idx の末尾に付けられる形にする
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        # torch.cat は同じ次元でテンソルをつなぐ。dim=1 は T の方向。[B, T] と [B, 1] → [B, T+1]
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
