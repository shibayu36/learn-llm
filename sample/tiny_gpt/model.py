import math

import torch
from torch import nn
from torch.nn import functional as F

from tiny_gpt.config import Config


def scaled_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_size = q.shape[-1]
    length = q.shape[-2]
    # 各位置のQueryと各位置のKeyの内積を一度に計算する。scores[b, i, j] は
    # 「位置iが探しているもの」と「位置jが持っているもの」の噛み合い度合い。
    # [B, T, D] @ [B, D, T] → [B, T, T]
    scores = q @ k.transpose(-2, -1)
    # 成分数が増えるほど内積の幅が広がるので、√D で割って幅を揃える。
    scores = scores / math.sqrt(key_size)

    if causal:
        # 各位置から見て未来にあたる位置だけTrueの表。対角線より右上が未来。[T, T]
        future = torch.ones(length, length, dtype=torch.bool)
        future = torch.triu(future, diagonal=1)
        # 未来の位置のスコアを −∞ にする。softmaxで exp(−∞) = 0 になり、未来への
        # 割合が0になる。予測すべき答えを見せないため。
        scores = scores.masked_fill(future, float("-inf"))

    # 行ごとに合計1の割合へ変換する。weights[b, i] は位置iが各位置を参照する割合。
    weights = torch.softmax(scores, dim=-1)
    # 各位置のValueをその割合で混ぜる。output[b, i] は位置iが集めた情報。[B, T, D]
    output = weights @ v
    return output, weights


class SelfAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 同じxから、探しているもの（Q）・持っているもの（K）・渡す中身（V）を作る。
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        mixed, weights = scaled_attention(q, k, v)
        # 集めた情報のうち、元のベクトルへどの成分をどれだけ書き足すかを学習で決める変換。
        output = self.output(mixed)
        return output, weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.expand = nn.Linear(d_model, 4 * d_model)
        self.contract = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # D個の成分を組み合わせて、4D個の特徴の候補を作る。各候補が「どの成分の
        # 組み合わせに反応するか」は学習で決まる。位置ごとに独立で、別の位置の
        # tokenは混ぜない。[B, T, D] → [B, T, 4D]
        hidden = self.expand(x)
        # 条件を満たさなかった候補（負の値）をほぼ0にし、反応した候補だけ残す。
        # これがないと2つのLinearは1つのLinearにまとまり、組み合わせの特徴を作れない。
        hidden = F.gelu(hidden)
        # 残った特徴を組み合わせてD次元へ戻す。1.9で元のベクトルへ足す更新分になる。
        # [B, T, 4D] → [B, T, D]
        output = self.contract(hidden)
        return output


class LayerNorm(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))
        self.epsilon = 0.00001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1つの位置のD個の成分について平均とばらつきを求め、平均0・ばらつき1に
        # 揃える。別の位置や別の系列とは混ぜない。平均・分散のshapeは [B, T, 1]
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        normalized = centered / torch.sqrt(variance + self.epsilon)
        # 揃えた後に、成分ごとの倍率（scale）とずらし（shift）を学習で掛け直す。
        return self.scale * normalized + self.shift


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attention = SelfAttention(d_model)
        self.norm2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 幅を揃えたxからAttentionで前の位置の情報を集める。2つ目の戻り値は
        # 観察用のAttention weightで、ここでは使わない。
        attention_output, _ = self.attention(self.norm1(x))
        # 置き換えず、元のベクトルに集めた情報を書き足す。
        x = x + attention_output
        # 集めた情報を含むベクトルからMLPで新しい特徴を作り、さらに書き足す。
        x = x + self.mlp(self.norm2(x))
        return x


# TinyGPTの構造
#
# ids [B, T]
#   │ token_embedding + position_embedding
#   ▼
# x [B, T, D]
#   │ transformer_block: TransformerBlock
#   │   x = x + attention(norm1(x))   SelfAttention・LayerNorm
#   │   x = x + mlp(norm2(x))         FeedForward・LayerNorm
#   ▼
# x [B, T, D]
#   │ final_norm: LayerNorm
#   │ lm_head: Linear(D, vocab_size)
#   ▼
# logits [B, T, vocab_size]
class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, config: Config) -> None:
        super().__init__()
        self.context_length = config.context_length
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.context_length, config.d_model
        )
        self.transformer_block = TransformerBlock(config.d_model)
        self.final_norm = LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        if length == 0 or length > self.context_length:
            raise ValueError("入力の長さがcontext_lengthの範囲外です")

        # 0, 1, 2, … という位置番号。位置の表の行を引くのに使う。
        positions = torch.arange(length)
        # token_embeddingは語彙数×Dの表で、各行が1つの文字の性質を表すベクトル。
        # idsの各文字IDでその文字の行を引き、整数のIDを学習で更新できるベクトルに
        # 置き換える。[B, T] → [B, T, D]
        x = self.token_embedding(ids)
        # 各位置のベクトルに、位置の表からその位置の行を足す。「何の文字か」に
        # 「何番目か」が重なったベクトルになる。位置の表は全系列で共通。
        # [B, T, D] + [T, D]
        x = x + self.position_embedding(positions)
        x = self.transformer_block(x)
        # 足し続けた本線のベクトルの幅を、内積を取る前に最後に一度揃える。
        x = self.final_norm(x)
        # 各位置のベクトルと、候補の文字ごとのベクトル（lm_headの重みの各行）との
        # 内積を取り、候補ごとの点数（logit）にする。[B, T, D] → [B, T, vocab_size]
        logits = self.lm_head(x)
        return logits
