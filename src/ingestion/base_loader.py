from abc import ABC, abstractclassmethod
import pandas as pd


class BaseNewsLoader(ABC):
    """
    Abstract base class for all news data sources.
    Any new source (NewsAPI, RSS, etc.) must implement these methods.
    """

    @abstractclassmethod
    def load(self) -> pd.DataFrame:
        """
        Load raw data from the source.
        Returns DataFrame with columns: [title, description, category]
        """
        pass

    @abstractclassmethod
    def get_source_name(self) -> str:
        """Return identifier string for this data source."""
        pass
