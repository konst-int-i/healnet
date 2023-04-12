from torch.utils.data import Dataset
from torchvision import transforms
from x_perceiver.utils import Config
from openslide import OpenSlide
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import *


def filter_manifest_files(config: Config, dataset: str):
    """
    Temporary util to filter the manifest files to only include the
    WSI images required
    Returns:
    """
    mol_df = TCGADataset(dataset, config).molecular_df
    manifest_path = Path(config.tcga_path).joinpath(f"gdc_manifests/full/{dataset}_wsi_manifest_full.txt")
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    manifest_filtered = manifest_df.loc[manifest_df.filename.isin(mol_df["slide_id"])]

    assert manifest_filtered.shape[0] == mol_df.shape[0], "Number of filtered manifest files does not match number of molecular files"

    write_path = Path(config.tcga_path).joinpath(f"gdc_manifests/filtered/{dataset}_wsi_manifest_filtered.txt")
    manifest_filtered.to_csv(write_path, sep="\t", index=False)
    print(f"Saved filtered manifest file to {write_path}")
    return None


class TCGADataset(Dataset):

    def __init__(self, dataset: str, config):
        self.dataset = dataset
        self.config = config
        self.molecular_df = self.load_molecular()


    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def get_info(self):
        pass

    def load_molecular(self) -> pd.DataFrame:
        data_path = Path(self.config.tcga_path).joinpath(f"molecular/tcga_{self.dataset}_all_clean.csv.zip")
        df = pd.read_csv(data_path, compression="zip", header=0, index_col=0, low_memory=False)

        # assign target column (high vs. low risk in equal parts of survival)
        target_col = "high_risk"
        df.loc[:, target_col] = pd.qcut(x=df["survival_months"], q=2, labels=[1, 0])

        return df

    def load_wsi(self, slide_id: str, level: int = None, resolution: str = None) -> Tuple:
        """
        Load in single slide and get region at specified resolution level
        Args:
            slide_id:
            level:
            resolution:

        Returns:
            Tuple (openslide object, tensor of region)
        """

        # load in openslide object
        slide_path = Path(self.config.tcga_path).joinpath(f"wsi/{self.dataset}/{slide_id}.svs")
        slide = OpenSlide(slide_path)

        # specify resolution level
        if resolution is None and level is None:
            raise ValueError("Must specify either resolution or level")
        elif resolution is not None:
            valid_resolutions = ["lowest", "mid", "highest"]
            assert resolution in valid_resolutions, f"Invalid resolution arg, must be one of {valid_resolutions}"
            if resolution == "lowest":
                level = slide.level_count - 1
            if resolution == "highest":
                level = 0
            if resolution == "mid":
                level = int(slide.level_count / 2)
            print("Loading slide region at level", level)

        # load in region
        size = slide.level_dimensions[level]
        region = slide.read_region((0,0), level, size)
        return slide, transforms.ToTensor()(region)



if __name__ == '__main__':
    os.chdir("../../")
    config = Config("config/main.yml").read()
    brca = TCGADataset("brca", config)
    blca = TCGADataset("blca", config)
    print(config)
    print(brca.molecular_df.shape, blca.molecular_df.shape)
    blca.load_wsi("TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F", resolution="lowest")

