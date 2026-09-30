from pathlib import Path

import huggingface_hub


# 日本語のSNS投稿にポジティブ・ネガティブのラベルが付いたデータ。
# 列は sentence・label・user_id・datetime で、label は 0 が positive、1 が negative。
SOURCE = "llm-book/wrime-sentiment"
# 配布元にはparquetがないので、Hugging Faceが自動変換したparquet
# （refs/convert/parquet ブランチ）の固定commitから取る。
REVISION = "e4d2dd3957122a481ca24ef4919bbd7239e8e7b9"
DATA_DIR = Path("data/wrime-sentiment")
SPLITS = ("train", "validation", "test")


def main() -> None:
    for split in SPLITS:
        local_path = huggingface_hub.hf_hub_download(
            repo_id=SOURCE, repo_type="dataset",
            filename="default/" + split + "/0000.parquet",
            revision=REVISION, local_dir=DATA_DIR,
        )
        print("取得しました:", local_path)


if __name__ == "__main__":
    main()
