import json
from pathlib import Path

import torch
import torch.nn as nn
import umap

from tokenizer import CharTokenizer

# 学習済みモデルの内部表現を観察するためのデータを作る。図はHTML側で描くので、
# ここでは投影した座標と元のベクトルをまとめるところまでを担当する

# PCAの軸を求めるのに使う、出現頻度上位の文字数。ほとんど学習で更新されない稀な文字が
# 軸を支配しないようにするため。投影自体は全tokenに対して行う
PCA_FIT_TOP_N = 300
# 出現回数がこれ未満の文字はUMAPの対象にしない。ほぼ初期値のままのランダムなベクトルで、
# 近傍関係に意味がないため
UMAP_MIN_COUNT = 100
# n_neighbors は「何個の近傍まで見て配置を決めるか」、min_dist は「近い点をどこまで詰めて
# 描くか」。この埋め込みは近傍関係が弱いので、小さい近傍を優先し詰めて描く設定にした。
# 既定の 15・0.1 と比べ、コサイン上位10が図の隣10に入る個数が平均1.4個から2.2個に増えた
UMAP_N_NEIGHBORS = 5
UMAP_MIN_DIST = 0.0


def count_token_frequencies(token_ids: list[int], vocab_size: int) -> list[int]:
    counts = [0] * vocab_size
    for token_id in token_ids:
        counts[token_id] += 1
    return counts


def fit_pca(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """PCA（主成分分析）で、点の散らばりが最も大きい方向から順に2本の軸を求める。

    vectors は [点の数, D]。返すのは平均 [D]、軸 [2, D]、各軸が全体の散らばりの
    何割を表しているか（寄与率）。
    """
    # 平均を引いて原点を点群の中心に移す。PCAは中心からの散らばりを見るため
    mean = vectors.mean(dim=0)
    centered = vectors - mean
    # torch.linalg.svd は行列を「回転 × 伸縮 × 回転」に分解する（特異値分解）。
    # vh の各行が散らばりの大きい順に並んだ軸、singular_values がその軸方向の伸び
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    axes = vh[:2]
    # 特異値の2乗が各軸方向の分散に比例するので、その割合が寄与率になる
    variances = singular_values ** 2
    explained_variance_ratio = (variances[:2] / variances.sum()).tolist()
    return mean, axes, explained_variance_ratio


def project(vectors: torch.Tensor, mean: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    # 中心を合わせてから各軸に射影する。[N, D] @ [D, 2] → [N, 2]
    return (vectors - mean) @ axes.T


def fit_umap(vectors: torch.Tensor, seed: int) -> torch.Tensor:
    """UMAPで [点の数, D] を [点の数, 2] にする。

    PCAと違って直線に射影するのではなく、元の空間で各点の近傍（n_neighbors個）を
    調べ、その近傍関係をできるだけ保つ2次元の配置を探す。近さはコサイン類似度で
    測り、近傍一覧と同じ基準にそろえる。random_state を固定すると毎回同じ配置になる
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=seed,
        n_jobs=1,
    )
    # UMAPはNumPy配列を受け取るので、テンソルを numpy() で変換して渡す
    coordinates = reducer.fit_transform(vectors.numpy())
    return torch.from_numpy(coordinates)


def build_embedding_map(
    model: nn.Module,
    tokenizer: CharTokenizer,
    train_token_ids: list[int],
    seed: int,
) -> dict:
    """token embeddingを2次元に投影し、全tokenの座標・頻度・元のベクトルをまとめる。

    PCAは頻度上位 PCA_FIT_TOP_N 文字で軸を求めて全tokenを投影する。UMAPは出現
    UMAP_MIN_COUNT 回以上の文字だけで計算し、それ以外の座標は None にする。
    tokens の添字がそのままtoken IDになる。
    """
    # detach() で学習用の計算グラフから切り離し、値だけのテンソルにする
    embeddings = model.tok_emb.weight.detach()
    counts = count_token_frequencies(train_token_ids, tokenizer.vocab_size)

    # 頻度の降順に並べたtoken ID。sorted の key に「その要素の並べ替えに使う値」を渡す
    ids_by_frequency = sorted(range(tokenizer.vocab_size), key=lambda token_id: -counts[token_id])
    mean, axes, explained_variance_ratio = fit_pca(embeddings[ids_by_frequency[:PCA_FIT_TOP_N]])
    pca_coordinates = project(embeddings, mean, axes)

    umap_ids: list[int] = []
    for token_id in range(tokenizer.vocab_size):
        if counts[token_id] >= UMAP_MIN_COUNT:
            umap_ids.append(token_id)
    umap_coordinates = fit_umap(embeddings[umap_ids], seed)
    # 全tokenぶんの座標リストを None で作り、UMAPを計算した文字だけ埋める
    umap_by_id: list[list[float] | None] = [None] * tokenizer.vocab_size
    for row in range(len(umap_ids)):
        umap_by_id[umap_ids[row]] = [
            round(umap_coordinates[row, 0].item(), 4),
            round(umap_coordinates[row, 1].item(), 4),
        ]

    tokens: list[dict] = []
    for token_id in range(tokenizer.vocab_size):
        vector: list[float] = []
        for value in embeddings[token_id].tolist():
            vector.append(round(value, 4))
        tokens.append({
            "token": tokenizer.int_to_str[token_id],
            "count": counts[token_id],
            "pca": [
                round(pca_coordinates[token_id, 0].item(), 4),
                round(pca_coordinates[token_id, 1].item(), 4),
            ],
            "umap": umap_by_id[token_id],
            "vector": vector,
        })

    return {
        "pca_fit_top_n": PCA_FIT_TOP_N,
        "explained_variance_ratio": explained_variance_ratio,
        "umap": {
            "min_count": UMAP_MIN_COUNT,
            "n_neighbors": UMAP_N_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
            "seed": seed,
        },
        "tokens": tokens,
    }


DATA_PLACEHOLDER = "/*__EMBEDDING_MAP_DATA__*/null"


def save_embedding_map_html(data: dict, template_path: Path, output_path: Path) -> None:
    """テンプレートHTMLにデータを埋め込み、ブラウザで開くだけで見られる1ファイルにする。"""
    template = template_path.read_text(encoding="utf-8")
    # </script> が文字列中に現れるとHTMLが壊れるので、< をエスケープしておく
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    output_path.write_text(template.replace(DATA_PLACEHOLDER, payload), encoding="utf-8")
