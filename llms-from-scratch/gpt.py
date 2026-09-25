import torch
import torch.nn as nn

from config import Config

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int) -> None:
        super().__init__()
        # 入力xをクエリ・キー・値の3種類のベクトルに射影する訓練可能な重み行列。
        # 情報検索の比喩で、クエリはいま注目しているトークン、キーはクエリと照合する
        # インデックス、値は取り出される実際の内容にあたる。
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        # 各位置が「どの位置を見てはいけないか」を表す表。行が今いる位置、列が参照先
        # の位置で、1が未来（見てはいけない）、0が自分と過去（見てよい）。
        # 「日本の首」の4文字なら次の表になる。
        #
        #            参照先→ 日  本  の  首
        #   位置0「日」      [0,  1,  1,  1]   「日」は自分しか見られない
        #   位置1「本」      [0,  0,  1,  1]   「本」は「日本」まで
        #   位置2「の」      [0,  0,  0,  1]
        #   位置3「首」      [0,  0,  0,  0]   最後の文字は全部見られる
        #
        # torch.triu(..., diagonal=1) は対角線より右上だけを残す（上三角）ので、
        # この形になる。forwardでは1の場所のスコアを -inf にしてAttentionの重みを
        # 0にする。実際の表は context_length × context_length で、入力の長さ分だけ
        # 切り出す。訓練で更新する値ではなく固定の表なので、parameterではなくbufferに
        # 登録する
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[1]
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # クエリとキーのドット積がAttentionスコア。ドット積はベクトルどうしの類似度の
        # 尺度で、大きいほど2つのトークンは類似していると見なされ、スコアも高くなる。
        # attn_scores[b, i, j] は位置iのクエリと位置jのキーの類似度。
        # [B, T, D] @ [B, D, T] → [B, T, T]
        attn_scores = queries @ keys.transpose(1, 2)
        # 未来の位置のスコアを -inf にする。softmaxで exp(-inf) = 0 になり、未来の
        # トークンの重みが0になる（Causal Attention）
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        # √D で割ってからsoftmaxで正規化し、行ごとに合計1のAttentionの重みにする。
        # 重みが大きいほど、その位置の重要度が高い。√D で割るのは、ドット積が大きく
        # なるとsoftmaxがステップ関数のようになり勾配が小さくなるのを避けるため。
        # 「日本の首」の位置2「の」の行なら（値は説明用）:
        #   score    日 0.8    本 1.4    の 0.3    首 -inf
        #   → exp      2.23      4.06      1.35      0        合計 7.63
        #   → ÷合計    0.29      0.53      0.18      0        合計 1。「の」は「本」を最も重視する
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        # 値ベクトルをAttentionの重みで加重和したものがコンテキストベクトル。
        # 位置2なら 0.29×V日 + 0.53×V本 + 0.18×Vの。他のトークンの情報を組み込んだ、
        # 強化された埋め込みベクトルになる。[B, T, T] @ [B, T, D] → [B, T, D]
        context_vec = attn_weights @ values
        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1つの位置のD個の成分を、平均0・分散1になるように調整する（層正規化）。
        # 別の位置とは混ぜない。D=3で x = [2, 4, 6] なら、平均4・分散2.67 → [-1.22, 0, 1.22]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # scaleとshiftは訓練可能なパラメータで、正規化後の値の倍率とずらしを学習で調整する
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 滑らかな活性化関数。負の値をほぼ0にし、正の値はほぼそのまま通す。
        #   x = -2 → -0.05,  -0.5 → -0.15,  0.5 → 0.35,  2 → 1.95
        # これがないと前後の2つのLinear層は1つの線形変換にまとまり、非線形な特徴を作れない
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi))
            * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # 埋め込み次元を4倍の高次元空間に拡張する
            GELU(),                           # 非線形変換を適用する
            nn.Linear(4 * d_model, d_model),  # 元の次元に縮小する
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 位置ごとに独立に適用され、他の位置のトークンは混ぜない。混ぜるのはAttentionだけ
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.att = CausalAttention(
            config.d_model, config.d_model, config.context_length
        )
        self.ff = FeedForward(config.d_model)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 層を深く積むと訓練が壊れる。以下の2つはその対策。
        # - ショートカット接続: 層の出力で入力を置き換えず、入力に足す。訓練では出口の誤差
        #   から各層の修正量（勾配）を入口へ逆向きに伝えるが、層ごとの掛け算で減衰して
        #   入口に届かなくなる。足し算にすると減衰しない直通路が残る
        # - 層正規化: 足し続けた値は層を重ねるほど桁が膨らむ。各部品の入口で平均0・分散1
        #   にそろえ、値の暴れによる訓練の不安定化を防ぐ
        # 各コンポーネントの前に層正規化を置く構成がPre-LayerNorm
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = x + shortcut

        # 同じ形で、フィードフォワードネットワークの出力を加算する
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x


# OneLayerOneHeadGPTの構造
#
# in_idx [B, T]
#   │ tok_emb + pos_emb
#   ▼
# x [B, T, D]
#   │ trf_block: x = x + att(norm1(x)); x = x + ff(norm2(x))
#   ▼
# x [B, T, D]
#   │ final_norm → out_head: Linear(D, vocab_size)
#   ▼
# logits [B, T, vocab_size]
class OneLayerOneHeadGPT(nn.Module):
    def __init__(self, vocab_size: int, config: Config) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)
        self.trf_block = TransformerBlock(config)
        self.final_norm = LayerNorm(config.d_model)
        self.out_head = nn.Linear(config.d_model, vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        seq_len = in_idx.shape[1]
        # トークン埋め込み: 語彙数×Dの表からトークンIDに対応する行を引き、IDを
        # 訓練可能な密ベクトルに変換する。[B, T] → [B, T, D]
        tok_embeds = self.tok_emb(in_idx)
        # 位置埋め込み: 0, 1, 2, … 番目に対応する行を引く。トークン埋め込みに加算する
        # ことで、「何のトークンか」に「何番目か」の情報が加わる。[T, D]
        pos_embeds = self.pos_emb(torch.arange(seq_len))
        x = tok_embeds + pos_embeds
        x = self.trf_block(x)
        # ショートカット接続で加算し続けたxを、最後にもう一度層正規化する
        x = self.final_norm(x)
        # 線形出力ヘッド。Transformerブロックの出力を語彙空間に射影し、語彙内のトークン
        # ごとにロジット（正規化されていない確率）を出す。[B, T, D] → [B, T, vocab_size]
        logits = self.out_head(x)
        return logits
