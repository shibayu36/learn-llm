import math
import torch
from torch import nn

from tiny_gpt.config import Config

def scaled_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_size  = q.shape[-1]
    # 各位置のQueryと各位置のKeyの内積を一度に計算する。scores[b, i, j] は
    # 「位置iが探しているもの」と「位置jが持っているもの」の噛み合い度合い。
    scores = q @ k.transpose(-2, -1)
    # 成分数が増えるほど内積の幅が広がるので、√D で割って幅を揃える。
    scores = scores / math.sqrt(key_size)
    # 行ごとに合計1の割合へ変換する。weight[b, i] は位置iが各位置を参照する割合。
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
        # 同じxから探しているもの（Q）・持っているもの（K）・渡す中身（V）を作る。
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        mixed, weights = scaled_attention(q, k, v)
        output = self.output(mixed)
        return output, weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1.7で実装するまで、入力をそのまま返す。
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1.8で実装するまで、入力をそのまま返す。
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1.9で実装するまで、入力をそのまま返す。
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
        # 文章の各tokenを、config.d_model次元のベクトルに変換
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        # 文章中の位置を表す
        self.position_embedding = nn.Embedding(
            self.context_length, config.d_model
        )
        self.transformer_block = TransformerBlock(config.d_model)
        self.final_norm = LayerNorm(config.d_model)
        # 1.10で、D次元からvocab_size次元へ変換するLinearに置き換える。
        self.lm_head = nn.Identity()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        if length == 0 or length > self.context_length:
            raise ValueError(f"入力の長さがcontext_lengthの範囲外です")

        # 0, 1, 2, ...という位置番号。位置の行を引くのに使う。
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
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x
