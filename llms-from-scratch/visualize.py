import json
from pathlib import Path

import torch
import torch.nn as nn
import umap

from gpt import TransformerBlock
from tokenizer import CharTokenizer

# 学習済みモデルの内部表現を観察するためのデータを作る。図や表はHTML側で描くので、
# ここでは投影した座標や集計した表をまとめるところまでを担当する
#
# 1. Token embeddingの地図: build_embedding_map
# 2. Hidden stateの表（予測と近い文脈）: build_hidden_state_tables

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


# ---- 2. Hidden stateの表 ----
#
# 入力文と対象位置は data/hidden-state-sentences.json。対象位置の文字を全文でそろえる
# ので、Layer 0（token埋め込み＋位置埋め込み）はどの文でも同じベクトルになり、層を
# 通ったあとの違いは前にある文脈の違いだけから生まれる

# 近い文脈を探す範囲。validationの先頭からこの数のtokenだけ使う。全層のベクトルを
# メモリに持つので全体は使わず、類似度0.9以上の近傍が十分に出る量にとどめた
NEIGHBOR_CORPUS_TOKENS = 40000
# 予測・近傍とも上位何件を出すか
TOP_K = 10


def collect_hidden_states(model: nn.Module, token_ids: list[int]) -> list[torch.Tensor]:
    """1つの入力を通し、Layer 0とTransformerBlockごとの出力を [T, D] のリストで返す。

    Layer 0 はブロックへの入力（token埋め込み＋位置埋め込み）、Layer i は i 番目の
    ブロックの出力。final_norm より前の値でそろえる。モデル側のコードは変えず、
    「モジュールの計算の前後に呼ばれる関数」（hook）を登録して途中の値を横取りする
    """
    blocks: list[nn.Module] = []
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            blocks.append(module)

    hidden_states: list[torch.Tensor] = []

    # inputs はforwardに渡された引数のタプル、output は戻り値。
    # どちらも [B, T, D] なので、先頭（B=1）を取り出して [T, D] にする
    def capture_block_input(module, inputs):
        hidden_states.append(inputs[0][0])

    def capture_block_output(module, inputs, output):
        hidden_states.append(output[0])

    handles = [blocks[0].register_forward_pre_hook(capture_block_input)]
    for block in blocks:
        handles.append(block.register_forward_hook(capture_block_output))

    # torch.no_grad() の中では学習用の計算グラフを作らないので、推論だけなら軽くなる
    with torch.no_grad():
        model(torch.tensor([token_ids]))
    for handle in handles:
        handle.remove()
    return hidden_states


def target_token(texts: list[str], position: int) -> str:
    """全文で position の文字が同じことを確かめて返す。違うと文脈だけの比較にならない。"""
    token = texts[0][position]
    for text in texts:
        if text[position] != token:
            raise ValueError(f"位置 {position} の文字が文によって違う: {text}")
    return token


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


def similarity_matrix(vectors: torch.Tensor) -> list[list[float]]:
    """[N, D] の各行どうしのコサイン類似度を N×N の表で返す。"""
    matrix: list[list[float]] = []
    for a in vectors:
        row: list[float] = []
        for b in vectors:
            row.append(round(cosine_similarity(a, b), 4))
        matrix.append(row)
    return matrix


def display_token(tokenizer: CharTokenizer, token_id: int) -> str:
    """表示・添字用に1文字へそろえる。

    <|endoftext|> や <|unk|> はそのままdecodeすると複数文字になり、前後の文字を
    切り出すときの添字がずれるので、専用の記号1文字に置き換える
    """
    if token_id == tokenizer.eos_id:
        return "␃"
    if token_id == tokenizer.unk_id:
        return "�"
    token = tokenizer.int_to_str[token_id]
    if token == "\n":
        return "␤"
    return token


def collect_neighbor_corpus(
    model: nn.Module, token_ids: list[int], context_length: int
) -> tuple[list[int], list[torch.Tensor]]:
    """近い文脈を探す土台を作る。

    context_length幅の非重複窓ごとにforwardし、層ごとに全窓のベクトルを
    [コーパスtoken数, D] にまとめ、コサイン類似度を内積で測れるよう各行を
    ノルムで割って正規化する。窓に満たない末尾のtokenは捨てる
    """
    corpus_token_ids: list[int] = []
    chunks_by_layer: list[list[torch.Tensor]] = []
    for start in range(0, len(token_ids) - context_length + 1, context_length):
        window = token_ids[start:start + context_length]
        layer_vectors = collect_hidden_states(model, window)
        if len(chunks_by_layer) == 0:
            for _ in layer_vectors:
                chunks_by_layer.append([])
        for layer in range(len(layer_vectors)):
            chunks_by_layer[layer].append(layer_vectors[layer])
        corpus_token_ids.extend(window)

    normalized_by_layer: list[torch.Tensor] = []
    for chunks in chunks_by_layer:
        matrix = torch.cat(chunks)
        normalized_by_layer.append(matrix / matrix.norm(dim=1, keepdim=True))
    return corpus_token_ids, normalized_by_layer


def predict_next_char(model: nn.Module, tokenizer: CharTokenizer, vector: torch.Tensor) -> list[dict]:
    """1つの層のベクトルから、そのまま出力層に通した次の文字の予測上位を返す（logit lens）。"""
    with torch.no_grad():
        logits = model.out_head(model.final_norm(vector))
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, TOP_K)
    predictions: list[dict] = []
    for k in range(TOP_K):
        predictions.append({
            "token": display_token(tokenizer, top.indices[k].item()),
            "prob": round(top.values[k].item(), 4),
        })
    return predictions


def find_neighbors(
    tokenizer: CharTokenizer,
    corpus_token_ids: list[int],
    normalized_corpus: torch.Tensor,
    vector: torch.Tensor,
) -> list[dict]:
    """コーパスの中から、コサイン類似度が高い文脈を前後の文字つきで返す。"""
    similarities = normalized_corpus @ (vector / vector.norm())
    top = torch.topk(similarities, TOP_K)
    neighbors: list[dict] = []
    for k in range(TOP_K):
        position = top.indices[k].item()
        before = ""
        for token_id in corpus_token_ids[max(0, position - 10):position]:
            before += display_token(tokenizer, token_id)
        after = ""
        for token_id in corpus_token_ids[position + 1:position + 4]:
            after += display_token(tokenizer, token_id)
        neighbors.append({
            "similarity": round(top.values[k].item(), 4),
            "before": before,
            "token": display_token(tokenizer, corpus_token_ids[position]),
            "after": after,
        })
    return neighbors


def build_hidden_state_tables(
    model: nn.Module,
    tokenizer: CharTokenizer,
    texts: list[str],
    target_positions: list[int],
    corpus_token_ids: list[int],
    context_length: int,
) -> dict:
    """対象位置・文・層ごとに、次の文字の予測上位（logit lens）と近い文脈の上位をまとめる。

    予測は各層のベクトルをfinal_norm→out_head→softmaxに通した確率の上位。
    近い文脈は、corpus_token_idsから作ったコーパスとのコサイン類似度の上位
    """
    corpus_tokens, normalized_corpus_by_layer = collect_neighbor_corpus(
        model, corpus_token_ids[:NEIGHBOR_CORPUS_TOKENS], context_length
    )

    hidden_states_by_text: list[list[torch.Tensor]] = []
    for text in texts:
        token_ids = tokenizer.encode(text)
        if tokenizer.unk_id in token_ids:
            raise ValueError(f"語彙にない文字を含む: {text}")
        # 対象位置ごとに forward し直すと無駄なので、文ごとに1回だけ通す
        hidden_states_by_text.append(collect_hidden_states(model, token_ids))

    num_layers = len(hidden_states_by_text[0])

    targets: list[dict] = []
    for position in target_positions:
        sentences: list[dict] = []
        vectors_by_sentence: list[torch.Tensor] = []
        for s in range(len(texts)):
            layer_vectors: list[torch.Tensor] = []
            for layer_states in hidden_states_by_text[s]:
                layer_vectors.append(layer_states[position])
            vectors_by_sentence.append(torch.stack(layer_vectors))

            layers: list[dict] = []
            for layer in range(num_layers):
                vector = layer_vectors[layer]
                layers.append({
                    "predictions": predict_next_char(model, tokenizer, vector),
                    "neighbors": find_neighbors(
                        tokenizer, corpus_tokens, normalized_corpus_by_layer[layer], vector
                    ),
                })
            sentence_text = texts[s]
            sentences.append({
                "text": sentence_text,
                "next_char": sentence_text[position + 1],
                "layers": layers,
            })

        # [文, 層, D] にまとめ、層ごとに文どうしのコサイン類似度を求める
        vectors = torch.stack(vectors_by_sentence)
        similarities: list[list[list[float]]] = []
        for layer in range(num_layers):
            similarities.append(similarity_matrix(vectors[:, layer]))

        targets.append({
            "position": position,
            "token": target_token(texts, position),
            "sentences": sentences,
            "similarities": similarities,
        })

    return {
        "num_layers": num_layers,
        "top_k": TOP_K,
        "corpus": {"num_tokens": len(corpus_tokens)},
        "targets": targets,
    }


# ---- HTMLへの埋め込み ----

DATA_PLACEHOLDER = "/*__DATA__*/null"


def save_html(data: dict, template_path: Path, output_path: Path) -> None:
    """テンプレートHTMLにデータを埋め込み、ブラウザで開くだけで見られる1ファイルにする。"""
    template = template_path.read_text(encoding="utf-8")
    # </script> が文字列中に現れるとHTMLが壊れるので、< をエスケープしておく
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    output_path.write_text(template.replace(DATA_PLACEHOLDER, payload), encoding="utf-8")
