import torch
import torch.nn as nn

from dataset import create_dataloader
from generate import generate, text_to_token_ids, token_ids_to_text
from tokenizer import CharTokenizer


MAX_NEW_TOKENS = 100


def evaluate_validation(
    model: nn.Module,
    token_ids: list[int],
    context_length: int,
    batch_size: int = 16,
) -> dict:
    """完全な窓すべてのlossをtoken数で平均する。"""
    loader = create_dataloader(
        token_ids,
        batch_size=batch_size,
        max_length=context_length,
        stride=context_length,
        shuffle=False,
        drop_last=False,
    )

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_batch, target_batch in loader:
            logits = model(input_batch)
            # 最後の小さいバッチも、他のバッチと同じ1tokenあたりの重みで集計する。
            loss_sum = nn.functional.cross_entropy(
                logits.flatten(0, 1), target_batch.flatten(), reduction="sum"
            )
            total_loss += loss_sum.item()
            total_tokens += target_batch.numel()

    loss = total_loss / total_tokens
    return {
        "loss": loss,
        "evaluated_tokens": total_tokens,
    }


def generate_samples(
    model: nn.Module,
    tokenizer: CharTokenizer,
    context_length: int,
    prompts: list[dict],
    temperature: float,
    seeds: list[int | None],
) -> list[dict]:
    """指定したtemperature・seedで各promptを生成し、promptと続きを別々に返す。"""
    model.eval()
    samples: list[dict] = []
    for entry in prompts:
        prompt = entry["prompt"]
        for seed in seeds:
            context_ids = text_to_token_ids(prompt, tokenizer)
            if seed is not None:
                torch.manual_seed(seed)
            token_ids = generate(
                model,
                context_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                context_size=context_length,
                temperature=temperature,
            )
            generated_ids = token_ids[:, context_ids.shape[1]:]
            samples.append({
                "prompt_id": entry["id"],
                "prompt": prompt,
                "temperature": temperature,
                "seed": seed,
                "generated_text": token_ids_to_text(generated_ids, tokenizer),
            })
    return samples
