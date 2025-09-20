from qlib.model.base import Model
from qlib.data.dataset import Dataset, DatasetH
import pandas as pd
from qlib.data import D

class SingleFactorModel(Model):
    """
    A custom model that calculates a factor based on a given formula and then
    generates a trading signal by selecting the top quantile of scores each day.
    """
    def __init__(self, factor_formula: str):
        """
        Initializes the SingleFactorModel.

        Args:
            factor_formula (str): The qlib-style formula for the alpha factor.
            quantile (float): The quantile of top stocks to select for the signal (e.g., 0.2 for top 20%).
        """
        self.factor_formula = factor_formula


    def fit(self, dataset: Dataset):
        # This is a simple factor model, so no training ("fitting") is required.
        pass

    def predict(self, dataset: Dataset, segment="test") -> pd.DataFrame:
        """
        Calculates the factor and generates the trading signal.
        This method is called by the SignalRecord.

        Args:
            dataset (Dataset): The dataset providing the context (instruments and time range).
            segment (str): The data segment to predict on (e.g., "test").

        Returns:
            pd.DataFrame: A DataFrame with a 'score' column containing the signal
                          for the top quantile of instruments each day.
        """
        # Determine the universe and time range from the dataset's segment
        df_index = dataset.prepare(segment).index
        instruments = df_index.get_level_values("instrument").unique()
        start_time, end_time = dataset.segments[segment]

        # 1. Calculate the raw factor scores using the provided formula
        factor_df = D.features(
            instruments,
            fields=[self.factor_formula],
            start_time=start_time,
            end_time=end_time,
        ).rename(columns={self.factor_formula: "score"})
        if factor_df.index.names[0] == 'instrument':
            factor_df = factor_df.swaplevel().sort_index()
        if factor_df.empty:
            print("Warning: Factor calculation resulted in an empty DataFrame.")
            return pd.DataFrame(columns=["score"])
        return factor_df
