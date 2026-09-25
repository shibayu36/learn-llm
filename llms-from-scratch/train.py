import torch
import torch.nn as nn


# 本の calc_loss_batch（deviceは使わない）。1バッチの交差エントロピー誤差（cross entropy loss）を
# 計算する。各位置で「正解の次の文字」にモデルが割り当てた確率 p を取り出し、-log(p) を全位置で
# 平均した値。正解に確率1を割り当てれば0、確率が低いほど大きくなる。
#   語彙4,052で確率が均等なら p = 1/4052 → -log(p) = 8.31。学習前はこの付近になる
#   正解の確率が 0.5 なら 0.69、0.9 なら 0.11
def calc_loss_batch(
    input_batch: torch.Tensor, target_batch: torch.Tensor, model: nn.Module
) -> torch.Tensor:
    logits = model(input_batch)
    # cross_entropy は [N, vocab_size] のロジットと [N] の正解IDを受け取り、softmax → 正解の
    # 確率を取り出す → log → 平均 → -1倍 をまとめて計算する。
    # flatten(0, 1) は B と T の次元をつぶして [B*T, vocab_size] に、flatten() は [B*T] にする
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss
