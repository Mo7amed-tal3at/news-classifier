import pandas as pd
from datasets import load_dataset
from src.ingestion.base_loader import BaseNewsLoader

LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


class AGNewsLoader(BaseNewsLoader):
    """
    Loads AG News dataset from Hugging Face.
    Drop-in replacement: swap this class with NewsAPILoader
    when API key is available — zero changes to the pipeline.
    """

    def __init__(self, split: str = "train"):
        """
        Args:
            split: "train" or "test"
        """
        self.split = split

    def get_source_name(self) -> str:
        return f"ag_news_{self.split}"

    def load(self) -> pd.DataFrame:
        print(f"[AGNewsLoader] Downloading AG News ({self.split} split)...")

        dataset = load_dataset("fancyzhx/ag_news", split=self.split)

        df = pd.DataFrame(
            {
                "title": dataset["text"],
                "description": "",  # AG News مفيهاش description منفصلة
                "category": [LABEL_MAP[label] for label in dataset["label"]],
                "source": self.get_source_name(),
            }
        )

        print(f"[AGNewsLoader] Loaded {len(df):,} articles.")
        return df
