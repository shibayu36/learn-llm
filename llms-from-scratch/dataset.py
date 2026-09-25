import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import CharTokenizer


def load_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            document = json.loads(line)
            texts.append(document["text"])
    return texts


# 全文書をトークンIDに変換し、文書と文書の間に <|endoftext|> を挟んで1本のID列にする。
# 本は文字列の段階で " <|endoftext|> " を挟んでからtiktokenでencodeするが、
# 文字Tokenizerでは "<|endoftext|>" が1文字ずつに分解されてしまうので、IDの段階で挟む
def join_texts_with_eos(texts: list[str], tokenizer: CharTokenizer) -> list[int]:
    token_ids: list[int] = []
    for text in texts:
        token_ids.extend(tokenizer.encode(text))
        token_ids.append(tokenizer.eos_id)
    return token_ids


# 本の GPTDatasetV1。1本のID列からスライディングウィンドウで max_length 個ずつ切り出し、
# 入力（input）と、それを1つ後ろにずらした正解（target）の組を作る。
#
# 「日本の首都は東京」を max_length=4、stride=4 で切ると次の2組になる。
#   input:  日 本 の 首      target:  本 の 首 都
#   input:  都 は 東 京      target:  は 東 京 (次の文字)
# 各位置の正解は「その位置の次の文字」なので、1組で max_length 個の予測問題を含む
class GPTDataset(Dataset):
    def __init__(self, token_ids: list[int], max_length: int, stride: int) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []
        # stride は窓を何個ずらすか。max_length と同じにすると窓どうしが重ならない
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            # torch.tensor はPythonのリストをテンソル（PyTorchの多次元配列）に変換する
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # DataLoader が呼ぶ2つのメソッド。__len__ は組の総数、__getitem__ は idx 番目の組を返す
    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


# 本の create_dataloader_v1。DataLoader は Dataset から batch_size 個ずつ取り出して
# [B, T] のテンソルにまとめる。shuffle=True なら取り出す順番を毎周ランダムにし、
# drop_last=True なら最後の端数のバッチを捨てる
def create_dataloader(
    token_ids: list[int],
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    dataset = GPTDataset(token_ids, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
