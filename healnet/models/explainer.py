from pathlib import Path
from box import Box
from torch.utils.data import DataLoader
from healnet.utils import unpickle
import torch
from typing import *
import pandas as pd

class Explainer(object):
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.config = unpickle(self.log_dir.joinpath("config.pkl"))
        self.test_data_indices = unpickle(self.log_dir.joinpath("test_data_indices.pkl"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(self.log_dir.joinpath("best_model.pt"), map_location=self.device)

        self.omic_df = self.load_omic_df()

        self.get_high_risk_patients()


    def load_omic_df(self):
        data_path = Path(self.config.tcga_path).joinpath(f"omic/tcga_{self.config.dataset}_all_clean.csv.zip")
        df = pd.read_csv(data_path, compression="zip", header=0, index_col=0, low_memory=False)
        return df


    def best_model(self, model):
        torch.save(model.state_dict(), self.log_dir.joinpath("best_model.pt"))





    def get_high_risk_patients(self, n: int=5) -> List[str]:
        """

        Args:
            n:

        Returns:
            List of slide IDs
        """
        x = 1
        filtered = self.omic_df.iloc[self.test_data_indices]

    def get_low_risk_patients(self, n: int=5) -> List[str]:
        """

        Args:
            n:

        Returns:
            List of slide IDs
        """


if __name__ == "__main__":
    log_path = "logs/blca_09-08-2023_17-36-36"

    e = Explainer(log_path)





