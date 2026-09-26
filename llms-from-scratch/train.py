import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from generate import generate, text_to_token_ids, token_ids_to_text
from tokenizer import CharTokenizer


# 本の calc_loss_batch。1バッチの交差エントロピー誤差（cross entropy loss）を
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


# 本のリスト5-2 calc_loss_loader。
# DataLoaderの先頭 num_batches バッチについて calc_loss_batch を計算し、その平均を返す。
# 評価は毎回同じ先頭 num_batches バッチで行う。訓練用のDataLoaderはshuffle=Trueなので、
# 評価には shuffle=False で作った別のDataLoaderを渡す。定規を毎回同じにするため
def calc_loss_loader(data_loader: DataLoader, model: nn.Module, num_batches: int) -> float:
    num_batches = min(num_batches, len(data_loader))
    total_loss = 0.0
    # enumerate は「添字と要素を同時に取り出す書き方」。i がバッチの通し番号
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model)
        total_loss += loss.item()
    return total_loss / num_batches


# 本の evaluate_model。train/validationそれぞれの先頭 num_batches バッチで
# 損失を測る。
#   model.eval() → model.train(): 訓練時だけ働く部品を止める・戻す切り替え。このモデルには
#     ドロップアウトなどの部品はないが、本にならって呼ぶ
#   torch.no_grad(): 勾配の記録を取らない。評価では勾配は不要なので、時間とメモリの節約になる
def evaluate_model(
    model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, num_batches: int
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, num_batches)
        val_loss = calc_loss_loader(validation_loader, model, num_batches)
    model.train()
    return train_loss, val_loss


# 本の generate_and_print_sample。start_contextの続きをgreedyで50文字生成して表示する。
# generate は内側で既に torch.no_grad() を使っているので、ここでは重ねて書かない
def generate_and_print_sample(
    model: nn.Module, tokenizer: CharTokenizer, context_length: int, start_context: str
) -> None:
    model.eval()
    context_ids = text_to_token_ids(start_context, tokenizer)
    token_ids = generate(
        model,
        context_ids,
        max_new_tokens=50,
        context_size=context_length,
        temperature=0.0,
    )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    # 改行が入ると表示が崩れるのでスペースに置き換える
    print(decoded_text.replace("\n", " "))
    model.train()


# 本のリスト5-3 train_model_simple に相当するが、epochではなく config.steps 回の更新で回す点が
# 違う。層数やヘッド数を変えた比較で「同じ回数だけ更新した」とそろえるため（plan.md）
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Config,
    tokenizer: CharTokenizer,
    start_context: str,
) -> tuple[list[int], list[float], list[float]]:
    steps_seen: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    # data_iter からバッチを順に取り出す。train_loaderの全バッチ（1,447バッチ）を取り出し
    # 終わると、次の next() で「もう要素がない」ことを示す StopIteration という例外が出る。
    # それを捕まえて iter(train_loader) を取り直すと、先頭から2周目に入る。trainは1,447バッチ
    # なので、2,000stepの学習では2周目に入る
    data_iter = iter(train_loader)
    # ループの step は「ここまでに更新した回数」。step 0 は学習前で、その時点の損失も記録する。
    # 最後の評価（step == config.steps）のあとは更新せずに抜けるので、range は steps + 1 まで回す
    for step in range(config.steps + 1):
        if step % config.eval_every == 0:
            train_loss, val_loss = evaluate_model(
                model, train_eval_loader, validation_loader, config.eval_batches
            )
            steps_seen.append(step)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"step {step:4d}: train loss {train_loss:.3f}, validation loss {val_loss:.3f}")
            generate_and_print_sample(model, tokenizer, config.context_length, start_context)
        if step == config.steps:
            break

        try:
            input_batch, target_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            input_batch, target_batch = next(data_iter)

        optimizer.zero_grad()  # 前のstepの勾配をリセットする
        loss = calc_loss_batch(input_batch, target_batch, model)
        loss.backward()  # 損失から全パラメータの勾配を計算する（誤差逆伝播）
        optimizer.step()  # 勾配を使ってパラメータを更新する

    return steps_seen, train_losses, val_losses
