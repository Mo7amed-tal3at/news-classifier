import os
from src.ingestion.ag_news_loader import AGNewsLoader
from src.ingestion.validators import run_all_validations


OUTPUT_DIR = "data/raw"


def run_ingestion(split: str = "train") -> str:
    loader = AGNewsLoader(split=split)
    df = loader.load()

    run_all_validations(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"ag_news_{split}.parquet")
    df.to_parquet(output_path, index=False)
    print(f"[Ingest] Saved to {output_path}")

    return output_path


if __name__ == "__main__":
    run_ingestion("train")
    run_ingestion("test")
