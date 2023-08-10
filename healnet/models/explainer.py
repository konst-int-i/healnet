from pathlib import Path
from box import Box
from torch.utils.data import DataLoader
from healnet.utils import unpickle
import torch
from typing import *
import pandas as pd

class Explainer(object):
    def __init__(self, log_dir: str, level: int = 2):
        self.log_dir = Path(log_dir)
        self.config = unpickle(self.log_dir.joinpath("config.pkl"))
        self.level = level
        self.dataset = self.config.dataset
        self.test_data_indices = unpickle(self.log_dir.joinpath("test_data_indices.pkl"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(self.log_dir.joinpath("best_model.pt"), map_location=self.device)

        self.omic_df = self.load_omic_df()

        high_risk = self.get_patients(risk="high")
        low_risk = self.get_patients(risk="low")


    def load_omic_df(self):

        data_path = Path(self.log_dir).joinpath(f"{self.dataset}_omic_overlap.csv.zip")
        df = pd.read_csv(data_path, compression="zip").drop("Unnamed: 0", axis=1)

        return df


    def load_slide(self, slide_id: str) -> torch.Tensor:



    def best_model(self, model):
        torch.save(model.state_dict(), self.log_dir.joinpath("best_model.pt"))

    def get_patients(self, n: int=5, risk: str = "high") -> List[str]:
        """

        Args:
            n:

        Returns:
            List of slide IDs
        """
        assert risk in ["high", "low"], "Invalid risk type"

        filtered = self.omic_df.iloc[self.test_data_indices]
        ascending = True if risk == "high" else False

        filtered.sort_values(by=["y_disc", "survival_months"], ascending=ascending, inplace=True) # TODO - check if ascending is correct

        high_risk_ids = filtered.iloc[:n]["slide_id"].tolist()
        return high_risk_ids



if __name__ == "__main__":
    # log_path = "logs/blca_09-08-2023_17-36-36"
    log_path = "logs/kirp_10-08-2023_18-31-30"

    e = Explainer(log_path)


