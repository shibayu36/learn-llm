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
    # [B, T, D] @ [B, D, T] → [B, T, T]。行iから列jを参照するスコア。
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(key_size)

    if causal:
        future = torch.ones(length, length, dtype=torch.bool, device=q.device)
        future = torch.triu(future, diagonal=1)
        # 未来のスコアを −∞ にすると、softmax後のAttention weightが0になる。
        scores = scores.masked_fill(future, float("-inf"))

    # 各行について参照先のAttention weightを合計1にし、その割合でValueを集める。
    weights = torch.softmax(scores, dim=-1)
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
        # 同じxから、参照する側の特徴Q、参照される側の特徴K、集める情報Vを作る。
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        mixed, weights = scaled_attention(q, k, v)
        output = self.output(mixed)
        return output, weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.expand = nn.Linear(d_model, 4 * d_model)
        self.contract = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linearは最後のD軸だけを変換する。別の位置のtokenは混ぜない。
        hidden = self.expand(x)
        hidden = F.gelu(hidden)
        output = self.contract(hidden)
        return output


class LayerNorm(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))
        self.epsilon = 0.00001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1tokenのD成分内で正規化する。平均のshapeは [B, T, 1]。
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        normalized = centered / torch.sqrt(variance + self.epsilon)
        return self.scale * normalized + self.shift


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attention = SelfAttention(d_model)
        self.norm2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_output, _ = self.attention(self.norm1(x))
        # 変換結果で置き換えず、元の表現へ更新分を足す。
        x = x + attention_output
        x = x + self.mlp(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, config: Config) -> None:
        super().__init__()
        self.context_length = config.context_length
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.context_length, config.d_model
        )
        self.block = TransformerBlock(config.d_model)
        self.final_norm = LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        if length == 0 or length > self.context_length:
            raise ValueError("入力の長さがcontext_lengthの範囲外です")

        positions = torch.arange(length, device=ids.device)
        # tokenの表現 [B, T, D] に、全系列で共通の位置表現 [T, D] を足す。
        x = self.token_embedding(ids)
        x = x + self.position_embedding(positions)
        x = self.block(x)
        x = self.final_norm(x)
        # 各位置から次token候補のlogitを出す。出力は [B, T, vocab_size]。
        logits = self.lm_head(x)
        return logits
