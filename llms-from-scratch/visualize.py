import json
from pathlib import Path

import torch
import torch.nn as nn
import umap

from gpt import TransformerBlock
from tokenizer import CharTokenizer

# 学習済みモデルの内部表現を観察するためのデータを作る。図はHTML側で描くので、
# ここでは投影した座標と元のベクトルをまとめるところまでを担当する
#
# 1. Token embeddingの地図: build_embedding_map
# 2. Hidden stateの軌跡: build_hidden_state_trajectories

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


# ---- 2. Hidden stateの軌跡 ----
#
# 入力文と対象位置は data/hidden-state-sentences.json。対象位置の文字を全文でそろえる
# ので、Layer 0（token埋め込み＋位置埋め込み）はどの文でも同じベクトルになり、層を
# 通ったあとの違いは前にある文脈の違いだけから生まれる


def collect_hidden_states(model: nn.Module, token_ids: list[int]) -> list[torch.Tensor]:
    """1つの入力を通し、Layer 0とTransformerBlockごとの出力を [T, D] のリストで返す。

    Layer 0 はブロックへの入力（token埋め込み＋位置埋め込み）、Layer i は i 番目の
    ブロックの出力。final_norm より前の値でそろえる。モデル側のコードは変えず、
    register_forward_hook で「そのモジュールが計算を終えるたびに呼ばれる関数」を
    登録して途中の値を横取りする
    """
    blocks: list[nn.Module] = []
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            blocks.append(module)

    hidden_states: list[torch.Tensor] = []
    handles = []
    for index in range(len(blocks)):
        is_first = index == 0

        def hook(module, inputs, output, is_first=is_first):
            # inputs はforwardに渡された引数のタプル、output は戻り値。
            # [B, T, D] の先頭（B=1）を取り出して [T, D] にする
            if is_first:
                hidden_states.append(inputs[0][0].detach())
            hidden_states.append(output[0].detach())

        handles.append(blocks[index].register_forward_hook(hook))

    # torch.no_grad() の中では学習用の計算グラフを作らないので、推論だけなら軽くなる
    with torch.no_grad():
        model(torch.tensor([token_ids]))
    for handle in handles:
        handle.remove()
    return hidden_states


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


def build_hidden_state_trajectories(
    model: nn.Module,
    tokenizer: CharTokenizer,
    texts: list[str],
    target_positions: list[int],
) -> dict:
    """対象tokenの各層のベクトルと、文どうしのコサイン類似度をまとめる。

    sentences[s]["targets"][t]["layers"][l] が、文 s の t 番目の対象位置の Layer l。
    similarities[t][l] は同じ対象位置・同じ層での文どうしの類似度行列
    """
    sentences: list[dict] = []
    for text in texts:
        token_ids = tokenizer.encode(text)
        if tokenizer.unk_id in token_ids:
            raise ValueError(f"語彙にない文字を含む: {text}")
        hidden_states = collect_hidden_states(model, token_ids)

        targets: list[dict] = []
        for position in target_positions:
            layers: list[dict] = []
            for layer_vectors in hidden_states:
                vector: list[float] = []
                for value in layer_vectors[position].tolist():
                    vector.append(round(value, 4))
                layers.append({
                    "norm": round(layer_vectors[position].norm().item(), 4),
                    "vector": vector,
                })
            targets.append({
                "position": position,
                "token": tokenizer.int_to_str[token_ids[position]],
                "layers": layers,
            })
        sentences.append({"text": text, "targets": targets})

    # 対象位置の文字が全文で同じでなければ、文脈の違いだけを見る前提が崩れる
    for target_index in range(len(target_positions)):
        first_token = sentences[0]["targets"][target_index]["token"]
        for sentence in sentences:
            if sentence["targets"][target_index]["token"] != first_token:
                raise ValueError(f"位置 {target_positions[target_index]} の文字が文によって違う")

    num_layers = len(sentences[0]["targets"][0]["layers"])
    similarities: list[list[list[list[float]]]] = []
    for target_index in range(len(target_positions)):
        per_layer: list[list[list[float]]] = []
        for layer in range(num_layers):
            matrix: list[list[float]] = []
            for a in sentences:
                row: list[float] = []
                vector_a = torch.tensor(a["targets"][target_index]["layers"][layer]["vector"])
                for b in sentences:
                    vector_b = torch.tensor(b["targets"][target_index]["layers"][layer]["vector"])
                    row.append(round(cosine_similarity(vector_a, vector_b), 4))
                matrix.append(row)
            per_layer.append(matrix)
        similarities.append(per_layer)

    return {
        "target_positions": target_positions,
        "num_layers": num_layers,
        "sentences": sentences,
        "similarities": similarities,
    }


def save_embedding_map_html(data: dict, template_path: Path, output_path: Path) -> None:
    """テンプレートHTMLにデータを埋め込み、ブラウザで開くだけで見られる1ファイルにする。"""
    template = template_path.read_text(encoding="utf-8")
    # </script> が文字列中に現れるとHTMLが壊れるので、< をエスケープしておく
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    output_path.write_text(template.replace(DATA_PLACEHOLDER, payload), encoding="utf-8")
