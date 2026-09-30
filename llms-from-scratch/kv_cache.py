import torch
import torch.nn as nn


# 1つのAttentionヘッドが計算したKとVを、トークンごとに溜めておく場所。
# 「日本の首都は」から生成すると、モデルにトークンを入れるたびに次のように溜まる。
#
#            入れるトークン   新しく計算するK・V    この後ここにあるK・V
#   1回目    日本の首都は     6トークン分           日本の首都は
#   2回目    東               「東」の分だけ        日本の首都は東
#   3回目    京               「京」の分だけ        日本の首都は東京
#
# 溜めておけるのは、後ろにトークンが増えても過去のトークンのK・Vが変わらないから。
# 「日本の首都は」を入れて計算した「は」のK・Vと、「日本の首都は東」を入れて計算した
# 「は」のK・Vは同じ値になる。
#
# Qは新しいトークンの分しか要らないので溜めない。2回目に求めたいのは「東」のコンテキスト
# ベクトルだけで、それは「東」のQと「日本の首都は東」7トークンのK・Vがあれば計算できる
class HeadKVCache:
    def __init__(self) -> None:
        # keys・values はどちらも [B, T_past, head_dim]（head_dim は CausalAttentionWithKVCache の D）。
        # まだ何も入っていなければ None
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    # 溜まっているトークン数。次に入れるトークンの位置（0始まり）でもある
    def num_tokens(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.shape[1]

    # 今回のトークンのK・Vを過去の分の後ろにつなぎ、過去と今回を合わせた全体を返す
    def append(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None or self.values is None:
            self.keys = keys
            self.values = values
        else:
            # torch.cat は複数のテンソルを指定した次元でつなげる。dim=1 はトークンの並び
            # （T）の次元なので、[B, T_past, head_dim] + [B, T_new, head_dim]
            # → [B, T_past + T_new, head_dim]
            self.keys = torch.cat([self.keys, keys], dim=1)
            self.values = torch.cat([self.values, values], dim=1)
        return self.keys, self.values


# モデル全体のKVキャッシュ。各ヘッドが自分のK・Vを作るので、層ごと・ヘッドごとに
# HeadKVCache を1つ持ち、blocks[層][ヘッド] で取り出す。同じモデルでも、入れたトークン列が
# 違えば中身は別になる
class KVCache:
    def __init__(self, model: nn.Module) -> None:
        self.blocks: list[list[HeadKVCache]] = []
        for block in model.trf_blocks:
            heads: list[HeadKVCache] = []
            for _ in block.att.heads:
                heads.append(HeadKVCache())
            self.blocks.append(heads)

    # 溜まっているトークン数。どの層・ヘッドにも同じ数が溜まっているので、先頭のヘッドの数を返す
    def num_tokens(self) -> int:
        return self.blocks[0][0].num_tokens()
