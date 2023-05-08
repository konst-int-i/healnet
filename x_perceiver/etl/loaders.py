import einops
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

    def __init__(self, dataset: str,
                 config: Box,
                 level: int=3,
                 filter_omic: bool = True,
                 survival_analysis: bool = True,
                 num_classes: int = 2,
                 n_bins: int = 4,
                 sources: List = ["omic", "slides"]):
        """
        Dataset wrapper to load different TCGA data modalities (omic and WSI data).
        Args:
            dataset:
            config:
            filter_omic: filter omic data (self.feature, self.molecular_df) to only include samples with
                corresponding WSI data
            n_bins: number of discretised bins for survival analysis


        Examples:
            >>> from x_perceiver.etl.loaders import TCGADataset
            >>> from x_perceiver.utils import Config
            >>> config = Config("config/main.yml").read()
            >>> dataset = TCGADataset("blca", config)
            # get omic data
            >>> dataset.omic_df
            # get sample slide
            >>> slide, tensor = dataset.load_wsi(blca.sample_slide_id, resolution="lowest")
            # get overall sample
            >>>
        """
        self.dataset = dataset
        self.sources = sources
        self.filter_omic = filter_omic
        self.survival_analysis = survival_analysis
        self.num_classes = num_classes
        self.n_bins = n_bins
        valid_sources = ["omic", "slides"]
        assert all([source in valid_sources for source in sources]), f"Invalid source specified. Valid sources are {valid_sources}"
        self.config = config
        self.data_conf = config.data
        self.wsi_paths: dict = self._get_slide_dict() # {slide_id: path}
        self.omic_df = self.load_omic()
        self.level = level
        self.slide_idx: dict = self._get_slide_idx() # {idx (molecular_df): slide_id}
        self.wsi_width, self.wsi_height = self.get_resize_dims(level=self.level)
        self.censorship = self.omic_df["censorship"].values
        self.survival_months = self.omic_df["survival_months"].values
        self.y_disc = self.omic_df["y_disc"].values
        self.features = self.omic_df.drop(["site", "oncotree_code", "case_id", "slide_id", "train", "censorship", "survival_months", "y_disc"], axis=1)
        self.get_info(full_detail=False)

    def __getitem__(self, index):
        slide_id = self.omic_df.iloc[index]["slide_id"]
        # # check that slide is available
        y_disc = self.y_disc[index]
        censorship = self.censorship[index]
        event_time = self.survival_months[index]
        if len(self.sources) == 1 and self.sources[0] == "omic":
            mol_tensor = torch.from_numpy(self.features.iloc[index].values)
            # introduce extra dim for perceiver
            mol_tensor = einops.repeat(mol_tensor, "feat -> b feat c", b=1, c=1)
            return mol_tensor.double(), censorship, event_time, y_disc
        elif len(self.sources) == 1 and self.sources[0] == "slides":
            slide, slide_tensor = self.load_wsi(slide_id, level=self.level)
            return slide_tensor, censorship, event_time, y_disc
        else: # both
            slide, slide_tensor = self.load_wsi(slide_id, level=self.level)
            mol_tensor = torch.from_numpy(self.features.iloc[index].values)
            return (mol_tensor, slide_tensor), censorship, event_time, y_disc

    def get_resize_dims(self, level: int):
        # TODO - validate which dimensions to pick for each level
        slide_sample = list(self.slide_idx.values())[0]
        slide_path = self.wsi_paths[slide_sample]
        slide = OpenSlide(slide_path)
        width = slide.level_dimensions[level][0]
        height = slide.level_dimensions[level][1]
        # take nearest multiple of 128 of height and width (for patches)
        width = round(width/128)*128
        height = round(height/128)*128
        return width, height

    def _get_slide_idx(self):
        # filter slide index to only include samples with WSIs available
        tmp_df = self.omic_df[self.omic_df.slide_id.isin(self.wsi_paths.keys())]
        return dict(zip(tmp_df.index, tmp_df["slide_id"]))

    def __len__(self):
        if self.sources == ["omic"]:
            # use all omic samples when running single modality
            return self.omic_df.shape[0]
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
        print(f"Molecular data shape: {self.omic_df.shape}")
        slide_dirs = [f for f in os.listdir(slide_path) if not f.startswith(".")]
        print(f"Total slides available: {len(slide_dirs)}")
        sample_overlap = (set(self.omic_df["slide_id"]) & set(self.wsi_paths.keys()))
        print(f"Molecular/Slide match: {len(sample_overlap)}/{len(self.omic_df)}")
        # print(f"Slide dimensions: {slide.dimensions}")
        print(f"Slide level count: {slide.level_count}")
        print(f"Slide level dimensions: {slide.level_dimensions}")
        print(f"Slide resize dimensions: w: {self.wsi_width}, h: {self.wsi_height}")
        print(f"Sources selected: {self.sources}")
        print(f"Censored share: {np.round(len(self.omic_df[self.omic_df['censorship'] == 1])/len(self.omic_df), 3)}")
        # print(f"Target column: {self.target_col}")

        if full_detail:
            pprint(dict(slide.properties))

    def show_samples(self, n=1):
        # sample_df = self.omic_df.sample(n=n)
        sample_df = self.omic_df[self.omic_df["slide_id"].isin(self.wsi_paths.keys())].sample(n=n)
        for idx, row in sample_df.iterrows():
            print(f"Case ID: {row['case_id']}")
            print(f"Patient age: {row['age']}")
            print(f"Gender: {'female' if row['is_female'] else 'male'}")
            print(f"Survival months: {row['survival_months']}")
            print(f"Survival years:  {np.round(row['survival_months']/12, 1)}")
            print(f"Censored (survived follow-up period): {'yes' if row['censorship'] else 'no'}")
            # print(f"Risk: {'high' if row['high_risk'] else 'low'}")
            # plot wsi
            slide, slide_tensor = self.load_wsi(row["slide_id"], level=self.level)
            print(f"Shape:", slide_tensor.shape)
            plt.figure(figsize=(10, 10))
            plt.imshow(slide_tensor)
            plt.show()




    def load_omic(self, eps: float = 1e-6) -> pd.DataFrame:
        data_path = Path(self.config.tcga_path).joinpath(f"omic/tcga_{self.dataset}_all_clean.csv.zip")
        df = pd.read_csv(data_path, compression="zip", header=0, index_col=0, low_memory=False)

        # filter samples for which there are no slides available
        if self.filter_omic:
            start_shape = df.shape[0]
            df = df[df["slide_id"].isin(self.wsi_paths.keys())]
            print(f"Filtered out {start_shape - df.shape[0]} samples for which there are no slides available")

        # assign target column (high vs. low risk in equal parts of survival)
        label_col = "survival_months"
        uncensored_df = df[df["censorship"] == 0]

        # take q_bins from uncensored patients only to determine the bins for all patients
        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=self.n_bins, retbins=True, labels=False)
        q_bins[-1] = df[label_col].max() + eps
        q_bins[0] = df[label_col].min() - eps

        # now take the bins for all patients
        df["y_disc"] = pd.cut(df[label_col], bins=q_bins, retbins=False, labels=False, right=False, include_lowest=True).values
        df["y_disc"] = df["y_disc"].astype(int)

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
    print(brca.omic_df.shape, blca.omic_df.shape)
    blca.load_wsi("TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F", resolution="lowest")

