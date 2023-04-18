from torch.utils.data import Dataset
from torchvision import transforms
from x_perceiver.utils import Config
from openslide import OpenSlide
import os
import torch
import pprint
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import *
from box import Box




class TCGADataset(Dataset):

    def __init__(self, dataset: str, config: Box, level: int=3, sources: List = ["molecular", "slides"]):
        """
        Dataset wrapper to load different TCGA data modalities (molecular and WSI data).
        Args:
            dataset:
            config:

        Examples:
            >>> from x_perceiver.etl.loaders import TCGADataset
            >>> from x_perceiver.utils import Config
            >>> config = Config("config/main.yml").read()
            >>> dataset = TCGADataset("blca", config)
            # get molecular data
            >>> dataset.molecular_df
            # get sample slide
            >>> slide, tensor = dataset.load_wsi(blca.sample_slide_id, resolution="lowest")
        """
        self.dataset = dataset
        self.sources = sources
        valid_sources = ["molecular", "slides"]
        assert all([source in valid_sources for source in sources]), f"Invalid source specified. Valid sources are {valid_sources}"
        self.config = config
        self.data_conf = config.data
        self.wsi_paths: dict = self._get_slide_dict() # {slide_id: path}
        self.molecular_df = self.load_molecular()
        self.level = level
        self.slide_idx: dict = self._get_slide_idx() # {idx (molecular_df): slide_id}
        self.wsi_width, self.wsi_height = self.get_resize_dims(level=self.level)
        self.target = self.molecular_df["high_risk"]
        self.features = self.molecular_df.drop(["site", "oncotree_code", "case_id", "slide_id"], axis=1)
        self.get_info(full_detail=False)

    def __getitem__(self, index):
        # TODO - implement cache
        slide_id = self.molecular_df.iloc[index]["slide_id"]
        # # check that slide is available
        label = self.target.iloc[index]
        if len(self.sources) == 1 and self.sources[0] == "molecular":
            mol_tensor = torch.from_numpy(self.features.iloc[index].values)
            return mol_tensor, label
        elif len(self.sources) == 1 and self.sources[0] == "slides":
            slide, slide_tensor = self.load_wsi(slide_id, level=self.level)
            return slide_tensor, label
        else: # both
            slide, slide_tensor = self.load_wsi(slide_id, level=self.level)
            mol_tensor = torch.from_numpy(self.features.iloc[index].values)
            return (mol_tensor, slide_tensor), label


    def get_resize_dims(self, level: int):
        # TODO - validate which dimensions to pick for each level
        print(self.slide_idx)
        slide_sample = list(self.slide_idx.values())[0]
        slide_path = self.wsi_paths[slide_sample]
        slide = OpenSlide(slide_path)
        print(slide.level_dimensions[level])
        width = slide.level_dimensions[level][0]
        height = slide.level_dimensions[level][1]
        # take nearest multiple of 128 of height and width (for patches)
        width = round(width/128)*128
        height = round(height/128)*128
        return width, height

    def _get_slide_idx(self):
        return dict(zip(self.molecular_df.index, self.molecular_df["slide_id"]))

    def __len__(self):
        if self.sources == ["molecular"]:
            # use all molecular samples when running single modality
            return self.molecular_df.shape[0]
        else:
            # only use overlap otherwise
            return len(self.wsi_paths)
    def _get_slide_dict(self):
        """
        Given the download structure of the gdc-client, each slide is stored in a folder
        with a non-meaningful name. This function returns a dictionary of slide_id to
        the path of the slide.
        Returns:
            svs_dict (dict): Dictionary of slide_id to path of slide
        """
        slide_path = Path(self.config.tcga_path).joinpath(f"wsi/{self.dataset}")
        svs_files = list(slide_path.glob("**/*.svs"))
        svs_dict = {path.name: path for path in svs_files}
        return svs_dict

    @property
    def sample_slide_id(self):
        return next(iter(self.wsi_paths.keys()))

    def get_info(self, full_detail: bool = False):
        slide, tensor = self.load_wsi(self.sample_slide_id, level=self.level)
        slide_path = Path(self.config.tcga_path).joinpath(f"wsi/{self.dataset}/")
        print(f"Dataset: {self.dataset.upper()}")
        print(f"Molecular data shape: {self.molecular_df.shape}")
        slide_dirs = [f for f in os.listdir(slide_path) if not f.startswith(".")]
        print(f"Total slides available: {len(slide_dirs)}")
        sample_overlap = (set(self.molecular_df["slide_id"]) & set(self.wsi_paths.keys()))
        print(f"Molecular/Slide match: {len(sample_overlap)}/{len(self.molecular_df)}")
        # print(f"Slide dimensions: {slide.dimensions}")
        print(f"Slide level count: {slide.level_count}")
        print(f"Slide level dimensions: {slide.level_dimensions}")
        print(f"Slide resize dimensions: w: {self.wsi_width}, h: {self.wsi_height}")
        print(f"Sources selected: {self.sources}")

        if full_detail:
            pprint(dict(slide.properties))


    def load_molecular(self) -> pd.DataFrame:
        data_path = Path(self.config.tcga_path).joinpath(f"molecular/tcga_{self.dataset}_all_clean.csv.zip")
        df = pd.read_csv(data_path, compression="zip", header=0, index_col=0, low_memory=False)

        # filter samples for which there are no slides available
        start_shape = df.shape[0]
        df = df[df["slide_id"].isin(self.wsi_paths.keys())]
        print(f"Filtered out {start_shape - df.shape[0]} samples for which there are no slides available")

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
        slide_path = self.wsi_paths[slide_id]
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
        if level > slide.level_count - 1:
            level = slide.level_count - 1
        # load in region
        size = slide.level_dimensions[level]
        region = slide.read_region((0,0), level, size)
        # print(f"Transforming image {slide_id}")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.wsi_height, self.wsi_width)),
            # RepeatTransform("c h w -> b c h w", b=1),
            RearrangeTransform("c h w -> h w c")
        ])

        return slide, transform(region)

class RearrangeTransform(object):
    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, img):
        img = rearrange(img, self.pattern)
        return img

class RepeatTransform(object):
    def __init__(self, pattern, b):
        self.pattern = pattern
        self.b = b
    def __call__(self, img):
        img = repeat(img, self.pattern, b=self.b)
        return img


if __name__ == '__main__':
    os.chdir("../../")
    config = Config("config/main.yml").read()
    brca = TCGADataset("brca", config)
    blca = TCGADataset("blca", config)
    print(config)
    print(brca.molecular_df.shape, blca.molecular_df.shape)
    blca.load_wsi("TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F", resolution="lowest")

