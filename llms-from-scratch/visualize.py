import json
from pathlib import Path

import torch
import torch.nn as nn
import umap

from gpt import CausalAttention, MultiHeadAttention, TransformerBlock
from tokenizer import CharTokenizer

# 学習済みモデルの内部表現を観察するためのデータを作る。図や表はHTML側で描くので、
# ここでは投影した座標や集計した表をまとめるところまでを担当する
#
# 1. Token embeddingの地図: build_embedding_map
# 2. Hidden stateの表（予測と近い文脈）: build_hidden_state_tables
# 3. ヘッドごとのAttention weight: build_attention_tables

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

# 予測・近傍とも上位何件を出すか
TOP_K = 10
# 近傍の「偶然の水準」。対象と別の文字の位置の類似度を高い順に並べ、上位 1/この値（0.1%）の
# 順位にある値を出す。上位10がこれに近ければ、同じくらい近い位置は他に数百ある。上位10はその中から
# たまたま選ばれただけなので、特にこの文脈に近いとは読まない。対象と同じ文字の位置を含めると、
# 「の」のように数万回出る文字では上位0.1%が同じ文字だけで埋まり、水準にならない
CHANCE_RANK_DIVISOR = 1000


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


def normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
    """各行をノルムで割り、コサイン類似度を内積で測れるようにする。"""
    return vectors / vectors.norm(dim=1, keepdim=True)


class TopK:
    """対象ベクトルごとの上位 k 件（類似度とコーパス上の位置）を、窓を見るたびに更新して持つ。

    search_neighbors がコーパスを窓ごとに見ながら、近い文脈の上位と偶然の水準を求めるのに使う。
    merge のたびにそれまでの上位と新しい窓を横に並べて上位 k 件を取り直すので、
    全位置の類似度を溜めなくてよい
    """

    def __init__(self, num_targets: int, k: int) -> None:
        self.k = k
        self.values = torch.full((num_targets, k), float("-inf"))
        self.positions = torch.zeros((num_targets, k), dtype=torch.long)

    def merge(self, similarities: torch.Tensor, positions: torch.Tensor) -> None:
        merged_values = torch.cat([self.values, similarities], dim=1)
        merged_positions = torch.cat([self.positions, positions], dim=1)
        top = torch.topk(merged_values, self.k, dim=1)
        self.values = top.values
        # gather は各行について、top.indices が指す列の値を merged_positions から拾う
        self.positions = torch.gather(merged_positions, 1, top.indices)


# 近い文脈は対象の文字との関係で3つに分けて出す。大きいモデルでは同じ単語の位置が上位を
# 埋めて他が見えなくなるため。文字Tokenizerなので、分類には直前の文字まで使う
#   same_bigram: 直前の文字も対象の文字も同じ。対象が銀行の「行」なら他の銀行の行、「の」なら他の「行の」。
#                対象が熟語の末尾なら、単語を認識できているかを読む
#   same_char:   対象の文字は同じで直前が違う。対象が銀行の「行」なら旅行・進行の行。
#                文字を共有する別の単語からどれだけ離れたかを読む
#   other_char:  対象の文字が違う。対象が銀行の「行」なら預金の金、融資の資。どの単語の仲間になったかを読む
NEIGHBOR_GROUPS = ["same_bigram", "same_char", "other_char"]


def search_neighbors(
    model: nn.Module,
    target_vectors_by_layer: list[torch.Tensor],
    target_token_ids: list[int],
    target_prev_token_ids: list[int],
    token_ids: list[int],
    context_length: int,
) -> tuple[list[int], int, list[TopK], dict[str, list[TopK]], dict[str, list[int]]]:
    """コーパスの全位置から、層ごとに各対象ベクトル（target_vectors_by_layer[layer] は [Q, D]）と
    類似度が高い位置を探す。

    context_length幅の非重複窓ごとにforwardし、その場で上位だけを残す。全層・全位置の
    ベクトルを溜めると12層・768次元では26GBになるため。分類には各対象の文字 target_token_ids と
    直前の文字 target_prev_token_ids を使う。窓に満たない末尾のtokenは捨てる。

    返すもの（括弧内はHTMLでの表示）:
    - 使ったtoken列（近傍の前後の文字を切り出す元）
    - chance_rank: 偶然の水準を取る順位。TOP_K と コーパス長 / CHANCE_RANK_DIVISOR の大きい方
    - 層ごとの「別の文字」の上位 chance_rank 件（末尾の値が文の下の「偶然の水準」）
    - NEIGHBOR_GROUPS の分類ごとの、層ごとの上位 TOP_K 件（3列の一覧）
    - 分類ごと・対象ごとの件数（列の先頭の件数）
    """
    num_windows = (len(token_ids) - context_length) // context_length + 1
    num_tokens = num_windows * context_length
    chance_rank = max(TOP_K, num_tokens // CHANCE_RANK_DIVISOR)
    num_layers = len(target_vectors_by_layer)
    num_targets = target_vectors_by_layer[0].shape[0]

    normalized_targets: list[torch.Tensor] = []
    for target_vectors in target_vectors_by_layer:
        normalized_targets.append(normalize_rows(target_vectors))

    chance: list[TopK] = []
    for _ in range(num_layers):
        chance.append(TopK(num_targets, chance_rank))
    by_group: dict[str, list[TopK]] = {}
    counts: dict[str, torch.Tensor] = {}
    for group in NEIGHBOR_GROUPS:
        by_group[group] = []
        for _ in range(num_layers):
            by_group[group].append(TopK(num_targets, TOP_K))
        counts[group] = torch.zeros(num_targets, dtype=torch.long)

    for start in range(0, num_tokens, context_length):
        window = token_ids[start:start + context_length]
        layer_vectors = collect_hidden_states(model, window)
        # この窓の各位置の、コーパス全体での位置番号。[T] を [Q, T] に広げる
        positions = torch.arange(start, start + context_length).expand(num_targets, -1)

        # 窓の先頭の直前の文字は前の窓の末尾。コーパス先頭には直前がないので -1
        window_tokens = torch.tensor(window)
        prev_of_first = token_ids[start - 1] if start > 0 else -1
        prev_tokens = torch.tensor([prev_of_first] + window[:-1])
        # 対象ごとに、この窓のどの位置が同じ文字か・同じ直前の文字かを [T] の真偽で求め、[Q, T] に積む
        same_char_rows: list[torch.Tensor] = []
        same_prev_rows: list[torch.Tensor] = []
        for q in range(num_targets):
            same_char_rows.append(window_tokens == target_token_ids[q])
            same_prev_rows.append(prev_tokens == target_prev_token_ids[q])
        same_char = torch.stack(same_char_rows)
        same_prev = torch.stack(same_prev_rows)
        masks = {
            "same_bigram": same_char & same_prev,
            "same_char": same_char & ~same_prev,
            "other_char": ~same_char,
        }
        for group in NEIGHBOR_GROUPS:
            counts[group] += masks[group].sum(dim=1)

        for layer in range(num_layers):
            # [Q, D] @ [D, T] → [Q, T]。対象ごとに、この窓の全位置との類似度
            similarities = normalized_targets[layer] @ normalize_rows(layer_vectors[layer]).T
            masked: dict[str, torch.Tensor] = {}
            for group in NEIGHBOR_GROUPS:
                # 分類に入らない位置の類似度を -inf にして、上位に選ばれないようにする
                masked[group] = similarities.masked_fill(~masks[group], float("-inf"))
                by_group[group][layer].merge(masked[group], positions)
            chance[layer].merge(masked["other_char"], positions)

    counts_as_lists: dict[str, list[int]] = {}
    for group in NEIGHBOR_GROUPS:
        counts_as_lists[group] = counts[group].tolist()
    return token_ids[:num_tokens], chance_rank, chance, by_group, counts_as_lists


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


def format_neighbors(
    tokenizer: CharTokenizer,
    corpus_token_ids: list[int],
    values: torch.Tensor,
    positions: torch.Tensor,
) -> list[dict]:
    """search_neighbors の1対象ぶんの結果から、上位 TOP_K 件を前後の文字つきで返す。

    該当する位置が TOP_K 件に満たないと類似度が -inf のまま残るので、そこで打ち切る
    """
    neighbors: list[dict] = []
    for k in range(TOP_K):
        if values[k].item() == float("-inf"):
            break
        position = positions[k].item()
        before = ""
        for token_id in corpus_token_ids[max(0, position - 10):position]:
            before += display_token(tokenizer, token_id)
        after = ""
        for token_id in corpus_token_ids[position + 1:position + 4]:
            after += display_token(tokenizer, token_id)
        neighbors.append({
            "similarity": round(values[k].item(), 4),
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
    近い文脈は、corpus_token_ids の全位置とのコサイン類似度の上位を NEIGHBOR_GROUPS で
    分けたものと、偶然の水準
    """
    token_ids_by_text: list[list[int]] = []
    hidden_states_by_text: list[list[torch.Tensor]] = []
    for text in texts:
        token_ids = tokenizer.encode(text)
        if tokenizer.unk_id in token_ids:
            raise ValueError(f"語彙にない文字を含む: {text}")
        token_ids_by_text.append(token_ids)
        # 対象位置ごとに forward し直すと無駄なので、文ごとに1回だけ通す
        hidden_states_by_text.append(collect_hidden_states(model, token_ids))

    num_layers = len(hidden_states_by_text[0])

    # 近傍探索はコーパスを1回通す間に全対象ベクトルを処理するので、対象位置×文のベクトルを
    # 層ごとに [Q, D] にまとめる。対象ベクトルの番号は q = 位置の添字 × 文の数 + 文の添字
    target_vector_lists: list[list[torch.Tensor]] = []
    for _ in range(num_layers):
        target_vector_lists.append([])
    target_token_ids: list[int] = []
    target_prev_token_ids: list[int] = []
    for position in target_positions:
        for s in range(len(texts)):
            for layer in range(num_layers):
                target_vector_lists[layer].append(hidden_states_by_text[s][layer][position])
            target_token_ids.append(token_ids_by_text[s][position])
            target_prev_token_ids.append(token_ids_by_text[s][position - 1] if position > 0 else -1)
    target_vectors_by_layer: list[torch.Tensor] = []
    for layer in range(num_layers):
        target_vectors_by_layer.append(torch.stack(target_vector_lists[layer]))
    corpus_tokens, chance_rank, chance, by_group, counts = search_neighbors(
        model, target_vectors_by_layer, target_token_ids, target_prev_token_ids,
        corpus_token_ids, context_length,
    )

    targets: list[dict] = []
    for p in range(len(target_positions)):
        position = target_positions[p]
        sentences: list[dict] = []
        for s in range(len(texts)):
            q = p * len(texts) + s
            layers: list[dict] = []
            for layer in range(num_layers):
                neighbors: dict[str, list[dict]] = {}
                for group in NEIGHBOR_GROUPS:
                    top = by_group[group][layer]
                    neighbors[group] = format_neighbors(
                        tokenizer, corpus_tokens, top.values[q], top.positions[q]
                    )
                layers.append({
                    "predictions": predict_next_char(
                        model, tokenizer, hidden_states_by_text[s][layer][position]
                    ),
                    "neighbors": neighbors,
                    "chance_similarity": round(chance[layer].values[q, chance_rank - 1].item(), 4),
                })
            neighbor_counts: dict[str, int] = {}
            for group in NEIGHBOR_GROUPS:
                neighbor_counts[group] = counts[group][q]
            sentence_text = texts[s]
            sentences.append({
                "text": sentence_text,
                "next_char": sentence_text[position + 1],
                "neighbor_counts": neighbor_counts,
                "layers": layers,
            })

        # 層ごとに文どうしのコサイン類似度を求める
        similarities: list[list[list[float]]] = []
        for layer in range(num_layers):
            vectors_of_layer: list[torch.Tensor] = []
            for s in range(len(texts)):
                vectors_of_layer.append(hidden_states_by_text[s][layer][position])
            similarities.append(similarity_matrix(torch.stack(vectors_of_layer)))

        targets.append({
            "position": position,
            "token": target_token(texts, position),
            "sentences": sentences,
            "similarities": similarities,
        })

    return {
        "num_layers": num_layers,
        "top_k": TOP_K,
        "neighbor_groups": NEIGHBOR_GROUPS,
        "corpus": {"num_tokens": len(corpus_tokens), "chance_rank": chance_rank},
        "targets": targets,
    }


# ---- 3. ヘッドごとのAttention weight ----
#
# CausalAttention.forwardはAttentionの重みを返さないので、hookでヘッドへの入力と出力を
# 横取りし、重みをここで計算し直す。計算し直した重みから作ったコンテキストベクトルが
# 実際の出力と一致することを確かめるので、gpt.pyの式が変わればここで気づける。


def collect_attention_heads(model: nn.Module) -> list[list[CausalAttention]]:
    """層ごとのヘッド（CausalAttention）の一覧を、層順・ヘッド順に返す。

    MultiHeadAttentionならその中の heads を1つずつ、CausalAttention単体（OneLayerOneHeadGPT・
    OneHeadGPT）ならそれ1つを、その層の唯一のヘッドとして扱う。
    """
    blocks: list[nn.Module] = []
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            blocks.append(module)

    heads_by_layer: list[list[CausalAttention]] = []
    for block in blocks:
        heads: list[CausalAttention] = []
        if isinstance(block.att, MultiHeadAttention):
            for head in block.att.heads:
                heads.append(head)
        elif isinstance(block.att, CausalAttention):
            heads.append(block.att)
        else:
            raise TypeError(f"未対応のAttention: {type(block.att)}")
        heads_by_layer.append(heads)
    return heads_by_layer


def collect_attention_head_io(
    model: nn.Module, heads_by_layer: list[list[CausalAttention]], token_ids: list[int]
) -> tuple[dict[CausalAttention, torch.Tensor], dict[CausalAttention, torch.Tensor]]:
    """1つの入力を通し、ヘッドごとに入力と出力を横取りして返す。pre_hookはforwardの前に
    入力を、forward_hookは後に出力を受け取る。

    入力は層正規化後の x で [T, d_model]、出力はコンテキストベクトルで [T, head_dim]。
    戻り値はヘッド（モジュール自身）をキーにした辞書（入力の辞書、出力の辞書）。
    """
    inputs_by_head: dict[CausalAttention, torch.Tensor] = {}
    outputs_by_head: dict[CausalAttention, torch.Tensor] = {}

    def capture_input(module: CausalAttention, inputs: tuple[torch.Tensor, ...]) -> None:
        inputs_by_head[module] = inputs[0][0]

    def capture_output(module: CausalAttention, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        outputs_by_head[module] = output[0]

    handles = []
    for heads in heads_by_layer:
        for head in heads:
            handles.append(head.register_forward_pre_hook(capture_input))
            handles.append(head.register_forward_hook(capture_output))

    with torch.no_grad():
        model(torch.tensor([token_ids]))
    for handle in handles:
        handle.remove()
    return inputs_by_head, outputs_by_head


def compute_attention_weights(head: CausalAttention, x: torch.Tensor) -> torch.Tensor:
    """1つのヘッドへの入力 x（[T, d_model]）からAttentionの重み（[T, T]）を計算する。

    gpt.py の CausalAttention.forward と同じ式（クエリとキーのドット積 → 未来をマスク →
    √D で割ってsoftmax）を、バッチ次元のない1文に対して書き直したもの
    """
    num_tokens = x.shape[0]
    queries = head.W_query(x)
    keys = head.W_key(x)

    attn_scores = queries @ keys.T  # [T, D] @ [D, T] → [T, T]
    mask = head.mask.bool()[:num_tokens, :num_tokens]
    attn_scores = attn_scores.masked_fill(mask, -torch.inf)
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    return attn_weights


def check_recomputed_output(
    head: CausalAttention,
    x: torch.Tensor,
    attn_weights: torch.Tensor,
    actual_output: torch.Tensor,
    layer: int,
    head_index: int,
) -> None:
    """計算し直した重みから作ったコンテキストベクトルが、実際の出力と一致するか確かめる。"""
    values = head.W_value(x)
    recomputed_output = attn_weights @ values
    # allclose は2つのテンソルの全要素が浮動小数点の誤差の範囲で等しいかを返す
    if not torch.allclose(recomputed_output, actual_output):
        raise ValueError(
            f"Layer {layer + 1} Head {head_index}: 計算し直したAttentionの出力が"
            "モデルの実際の出力と一致しない"
        )


def sum_attention_patterns(attn_weights: torch.Tensor) -> dict[str, float]:
    """1つの [T, T] 行列（1文分）から、位置1以降の prev・first・distance の合計を返す。

      - prev: 直前の文字への重み w[i, i-1] の合計
      - first: 先頭の文字への重み w[i, 0] の合計
      - distance: 参照距離 sum_{j<=i} w[i, j] * (i - j) の合計
    """
    num_tokens = attn_weights.shape[0]
    prev_sum = 0.0
    first_sum = 0.0
    distance_sum = 0.0
    for i in range(1, num_tokens):
        prev_sum += attn_weights[i, i - 1].item()
        first_sum += attn_weights[i, 0].item()
        distance = 0.0
        for j in range(i + 1):
            distance += attn_weights[i, j].item() * (i - j)
        distance_sum += distance
    return {"prev": prev_sum, "first": first_sum, "distance": distance_sum}


def build_attention_tables(model: nn.Module, tokenizer: CharTokenizer, texts: list[str]) -> dict:
    """文ごと・層ごと・ヘッドごとにAttentionの重みを求め、表とヒートマップ用にまとめる。

    流れ: hookで入出力を集める → 重みを計算し直す → 自己チェック → 要約値を足し込む →
    JSON用の行列にする。

    要約値（層×ヘッドごと）は全文・全位置（位置0は自分しか見られないので除く）の平均。
      - prev: 直前の文字への重み w[i, i-1]
      - first: 先頭の文字への重み w[i, 0]
      - distance: 参照距離の平均 sum_j w[i, j] * (i - j)
    位置1では直前の文字と先頭の文字が同じなので、prev と first に同じ重みが入る。

    "uniform" は比較用の基準値で、位置 i（i >= 1）が自分より前の i+1 個の位置を均等に
    見た場合の prev・first・distance（全文・全位置の平均）。
    """
    heads_by_layer = collect_attention_heads(model)
    num_layers = len(heads_by_layer)
    num_heads = len(heads_by_layer[0])

    pattern_sums: list[list[dict[str, float]]] = []
    for layer in range(num_layers):
        row: list[dict[str, float]] = []
        for head_index in range(num_heads):
            row.append({"prev": 0.0, "first": 0.0, "distance": 0.0})
        pattern_sums.append(row)
    position_count = 0
    uniform_prev_first_sum = 0.0
    uniform_distance_sum = 0.0

    sentences: list[dict] = []
    for text in texts:
        token_ids = tokenizer.encode(text)
        if tokenizer.unk_id in token_ids:
            raise ValueError(f"語彙にない文字を含む: {text}")
        num_tokens = len(token_ids)

        inputs_by_head, outputs_by_head = collect_attention_head_io(model, heads_by_layer, token_ids)

        weights_by_layer: list[list[list[list[float]]]] = []
        for layer in range(num_layers):
            weights_by_head: list[list[list[float]]] = []
            for head_index in range(num_heads):
                head = heads_by_layer[layer][head_index]
                x = inputs_by_head[head]
                attn_weights = compute_attention_weights(head, x)
                check_recomputed_output(head, x, attn_weights, outputs_by_head[head], layer, head_index)

                patterns = sum_attention_patterns(attn_weights)
                pattern_sums[layer][head_index]["prev"] += patterns["prev"]
                pattern_sums[layer][head_index]["first"] += patterns["first"]
                pattern_sums[layer][head_index]["distance"] += patterns["distance"]

                rows: list[list[float]] = []
                for i in range(num_tokens):
                    row: list[float] = []
                    for j in range(num_tokens):
                        row.append(round(attn_weights[i, j].item(), 4))
                    rows.append(row)
                weights_by_head.append(rows)
            weights_by_layer.append(weights_by_head)
        position_count += num_tokens - 1

        for i in range(1, num_tokens):
            # 全位置を均等に見た場合: i+1個の位置に等しい重み 1/(i+1)、参照距離の平均は
            # 0, 1, ..., i の平均で i/2
            uniform_prev_first_sum += 1.0 / (i + 1)
            uniform_distance_sum += i / 2.0

        chars: list[str] = []
        for token_id in token_ids:
            chars.append(display_token(tokenizer, token_id))
        sentences.append({"text": text, "chars": chars, "weights": weights_by_layer})

    summary: list[list[dict]] = []
    for layer in range(num_layers):
        row: list[dict] = []
        for head_index in range(num_heads):
            patterns = pattern_sums[layer][head_index]
            row.append({
                "prev": round(patterns["prev"] / position_count, 4),
                "first": round(patterns["first"] / position_count, 4),
                "distance": round(patterns["distance"] / position_count, 4),
            })
        summary.append(row)

    uniform = {
        "prev": round(uniform_prev_first_sum / position_count, 4),
        "first": round(uniform_prev_first_sum / position_count, 4),
        "distance": round(uniform_distance_sum / position_count, 4),
    }

    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "sentences": sentences,
        "summary": summary,
        "uniform": uniform,
    }


# ---- HTMLへの埋め込み ----

DATA_PLACEHOLDER = "/*__DATA__*/null"


def save_html(data: dict, template_path: Path, output_path: Path) -> None:
    """テンプレートHTMLにデータを埋め込み、ブラウザで開くだけで見られる1ファイルにする。"""
    template = template_path.read_text(encoding="utf-8")
    # </script> が文字列中に現れるとHTMLが壊れるので、< をエスケープしておく
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    output_path.write_text(template.replace(DATA_PLACEHOLDER, payload), encoding="utf-8")
