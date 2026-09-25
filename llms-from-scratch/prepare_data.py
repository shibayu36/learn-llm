import hashlib
import json
import random
from pathlib import Path

import huggingface_hub
import pyarrow
import pyarrow.parquet as parquet


SOURCE = "hotchpotch/fineweb-2-edu-japanese"
REVISION = "180ca004c6a89b590daaad86cb062a07a5353c69"
SUBSET = "small_tokens_cleaned"
SOURCE_FILE = "small_tokens_cleaned/train-00000-of-00283.parquet"
DATA_DIR = Path("data/fineweb-japanese-10k")
DOCUMENT_COUNT = 10000
VALIDATION_COUNT = 1000
SEED = 42


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_documents(path: Path, documents: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for document in documents:
            output.write(json.dumps(document, ensure_ascii=False) + "\n")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for split in ("train", "validation"):
            actual = file_sha256(DATA_DIR / (split + ".jsonl"))
            if actual != manifest["splits"][split]["sha256"]:
                raise ValueError("固定データの内容が変わっています: " + split)
        print("保存済みの固定データを使います:", DATA_DIR)
        return

    print("固定revisionの最初のファイルを取得します:", REVISION, flush=True)
    local_path = huggingface_hub.hf_hub_download(
        repo_id=SOURCE, repo_type="dataset", filename=SOURCE_FILE,
        revision=REVISION,
    )
    documents: list[dict[str, str]] = []
    seen_texts: set[str] = set()
    skipped_empty = 0
    skipped_duplicates = 0
    source_row = 0
    with parquet.ParquetFile(local_path) as source:
        for batch in source.iter_batches(
            batch_size=1000, columns=["id", "text", "url"], use_threads=False
        ):
            for example in batch.to_pylist():
                source_row += 1
                # 配布元が案内している、先頭10,000件と次の10,000件の重複を除く。
                if source_row <= 10000:
                    continue
                text = example["text"].strip()
                if not text:
                    skipped_empty += 1
                    continue
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if text_hash in seen_texts:
                    skipped_duplicates += 1
                    continue
                seen_texts.add(text_hash)
                documents.append({
                    "id": example["id"],
                    "text": text,
                    "url": example["url"],
                })
                if len(documents) % 1000 == 0:
                    print("取得した文書:", len(documents), flush=True)
                if len(documents) == DOCUMENT_COUNT:
                    break
            if len(documents) == DOCUMENT_COUNT:
                break
    if len(documents) != DOCUMENT_COUNT:
        raise ValueError("最初のファイルから必要な文書数を取得できませんでした")

    random.Random(SEED).shuffle(documents)
    splits: dict[str, list[dict[str, str]]] = {
        "train": documents[VALIDATION_COUNT:],
        "validation": documents[:VALIDATION_COUNT],
    }
    manifest: dict[str, object] = {
        "source": SOURCE,
        "revision": REVISION,
        "subset": SUBSET,
        "source_split": "train",
        "skipped_source_rows": 10000,
        "skipped_empty": skipped_empty,
        "skipped_exact_duplicates": skipped_duplicates,
        "selection": "first 10000 unique nonempty stripped texts after skip",
        "split_seed": SEED,
        "source_file": SOURCE_FILE,
        "huggingface_hub_version": huggingface_hub.__version__,
        "pyarrow_version": pyarrow.__version__,
    }
    split_report: dict[str, dict[str, int | str]] = {}
    for name, rows in splits.items():
        path = DATA_DIR / (name + ".jsonl")
        write_documents(path, rows)
        characters = 0
        for row in rows:
            characters += len(row["text"])
        split_report[name] = {
            "documents": len(rows),
            "characters": characters,
            "sha256": file_sha256(path),
        }
    manifest["splits"] = split_report
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(split_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
