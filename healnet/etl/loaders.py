import einops
from torch.utils.data import Dataset
from torchvision import transforms
from healnet.utils import Config
from openslide import OpenSlide
import os
from multiprocessing import Lock
from multiprocessing import Manager
import h5py
import torch
import pprint
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import *
from box import Box


class MMDataset(Dataset):
    """
    Generic torch dataset object for supervised multi-modal data.
    """
    def __init__(self, tensors: List[torch.Tensor], target: Optional[torch.Tensor]=None):
        """
        Args:
            tensors(List[torch.Tensor]): modalities for each sample
            target(torch.Tensor): label for each sample
        """
        self.tensors = tensors
        self.target = target

    def __getitem__(self, idx) -> [Tuple[List[torch.Tensor], torch.Tensor], List[torch.Tensor]]:
        if self.target is None:
            return [t[idx] for t in self.tensors]
        else:
            return [t[idx] for t in self.tensors], self.target[idx]

    def __len__(self):
        return self.tensors[0].size()[0]


class TCGADataset(Dataset):
    """
    Main dataset class for TCGA data. Loads in omic data and WSI data and returns a tuple of tensors when
    __getitem__ is called along with survival information (censorship, event_time, discretised survival).
    """

    def __init__(self, dataset: str,
                 config: Box,
                 level: int=2,
                 filter_overlap: bool = True,
                 survival_analysis: bool = True,
                 num_classes: int = 2,
                 n_bins: int = 4,
                 sources: List = ["omic", "slides"],
                 log_dir = None,
                 ):
        """
        Dataset wrapper to load different TCGA data modalities (omic and WSI data).
        Args:
            dataset (str): TCGA dataset to load (e.g. "brca", "blca")
            config (Box): Config object
            filter_overlap: filter omic data and/or slides that do not have a corresponding sample in the other modality
            n_bins: number of discretised bins for survival analysis

        Examples:
            >>> from healnet.etl.loaders import TCGADataset
            >>> from healnet.utils import Config
            >>> config = Config("config/main.yml").read()
            >>> dataset = TCGADataset("blca", config)
            # get omic data
            >>> dataset.omic_df
            # get sample slide
            >>> slide, tensor = dataset.load_omic(blca.sample_slide_id, resolution="lowest")
            # get overall sample
            >>> (slide, tensor), censorship, event_time, y_disc = next(iter(dataset))
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset
        self.log_dir = log_dir
        self.sources = sources
        self.filter_overlap = filter_overlap
        self.survival_analysis = survival_analysis
        self.sample_missing = False
        self.num_classes = num_classes
        self.n_bins = n_bins
        self.subset = self.config["survival.subset"]
        self.raw_path = Path(config.tcga_path).joinpath(f"wsi/{dataset}")
        prep_path = Path(config.tcga_path).joinpath(f"wsi/{dataset}_preprocessed_level{level}")
        self.prep_path = prep_path
        # create patch feature directory for first-time run
        os.makedirs(self.prep_path.joinpath("patch_features"), exist_ok=True)
        self.slide_ids = [slide_id.rsplit(".", 1)[0] for slide_id in os.listdir(prep_path.joinpath("patches"))]



        # for early fusion baseline, we need to concatenate omic and slide features into a single tensor
        self.concat = True if self.config.model in ["fcnn", "healnet_early"] and len(self.sources) > 1 else False

        valid_sources = ["omic", "slides"]
        assert all([source in valid_sources for source in sources]), f"Invalid source specified. Valid sources are {valid_sources}"
        self.wsi_paths: dict = self._get_slide_dict() # {slide_id: path}
        self.sample_slide_id = self.slide_ids[0] + ".svs"
        self.sample_slide = OpenSlide(self.wsi_paths[self.sample_slide_id])
        # pre-load and transform omic data
        self.omic_df = self.load_omic()
        self.features = self.omic_df.drop(["site", "oncotree_code", "case_id", "slide_id", "train", "censorship", "survival_months", "y_disc"], axis=1)
        self.omic_tensor = torch.Tensor(self.features.values)
        if self.config.model in ["healnet", "healnet_early"]:
            # Healnet expects inputs of the shape (batch_size, input_dim, channels)
            if self.config.omic_attention:
                self.omic_tensor = einops.repeat(self.omic_tensor, "n feat -> n channels feat", channels=1)
            else:
                self.omic_tensor = einops.repeat(self.omic_tensor, "n feat -> n feat channels", channels=1)


        self.level = level
        self.slide_idx: dict = self._get_slide_idx() # {idx (molecular_df): slide_id}
        self.wsi_width, self.wsi_height = self.get_resize_dims(level=self.level, override=config["data.resize"])
        self.censorship = self.omic_df["censorship"].values
        self.survival_months = self.omic_df["survival_months"].values
        self.y_disc = self.omic_df["y_disc"].values

        manager = Manager()
        self.patch_cache = manager.dict()
        # self.patch_cache = SharedLRUCache(capacity=256) # capacity should be multiple of num_workers
        print(f"Dataloader initialised for {dataset} dataset")
        self.get_info(full_detail=False)

    def __getitem__(self, index):
        y_disc = self.y_disc[index]
        censorship = self.censorship[index]
        event_time = self.survival_months[index]


        if len(self.sources) == 1 and self.sources[0] == "omic":
            omic_tensor = self.omic_tensor[index]
            return [omic_tensor], censorship, event_time, y_disc

        elif len(self.sources) == 1 and self.sources[0] == "slides":
            slide_id = self.omic_df.iloc[index]["slide_id"].rsplit(".", 1)[0]

           
            if index not in self.patch_cache:
                slide_tensor = self.load_patch_features(slide_id)
                self.patch_cache[index] = slide_tensor

            else:
                slide_tensor = self.patch_cache[index]
            if self.config.model == "fcnn": # for fcnn baseline
                slide_tensor = torch.flatten(slide_tensor)

            return [slide_tensor], censorship, event_time, y_disc

        else: # both
            omic_tensor = self.omic_tensor[index]
            slide_id = self.omic_df.iloc[index]["slide_id"].rsplit(".", 1)[0]

            if index not in self.patch_cache:
                slide_tensor = self.load_patch_features(slide_id)
                self.patch_cache[index] = slide_tensor
            else:
                slide_tensor = self.patch_cache[index]

            if self.concat: # for early fusion baseline
                slide_flat = torch.flatten(slide_tensor)
                omic_flat = torch.flatten(omic_tensor)
                concat_tensor = torch.cat([omic_flat, slide_flat], dim=0)
                if self.config.model == "healnet_early":
                    concat_tensor = concat_tensor.unsqueeze(0)
                return [concat_tensor], censorship, event_time, y_disc
            else: # keep separate for HEALNet
                return [omic_tensor, slide_tensor], censorship, event_time, y_disc

    def get_resize_dims(self, level: int, patch_height: int = 128, patch_width: int = 128, override=False):
        # TODO - use TIA to handle resizing
        if override is False:
            width = self.sample_slide.level_dimensions[level][0]
            height = self.sample_slide.level_dimensions[level][1]
            # take nearest multiple of 128 of height and width (for patches)
            width = round(width/patch_width)*patch_width
            height = round(height/patch_height)*patch_height
        else:
            width = self.config["data.resize_width"]
            height = self.config["data.resize_height"]
        return width, height

    def _get_slide_idx(self):
        # filter slide index to only include samples with WSIs availables
        filter_keys = [slide_id + ".svs" for slide_id in self.slide_ids]
        tmp_df = self.omic_df[self.omic_df.slide_id.isin(filter_keys)]
        return dict(zip(tmp_df.index, tmp_df["slide_id"]))

    def __len__(self):
        if self.sources == ["omic"]:
            # use all omic samples when running single modality
            return self.omic_df.shape[0]
        else:
            # only use overlap otherwise
            return len(self.slide_ids)
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

    # def _load_patch_coords(self):
    #     """
    #     Loads all patch coordinates for the dataset and level specified in the config and writes it to a dictionary
    #     with key: slide_id and value: patch coordinates (where each coordinate is a x,y tupe)
    #     """
    #     coords = {}
    #     for slide_id in self.slide_ids:
    #         patch_path = self.prep_path.joinpath(f"patches/{slide_id}.h5")
    #         h5_file = h5py.File(patch_path, "r")
    #         patch_coords = h5_file["coords"][:]
    #         coords[slide_id] = patch_coords
    #     return coords

    def get_info(self, full_detail: bool = False):
        """
        Logging util to print some basic dataset information. Normally called at the start of a pipeline run
        Args:
            full_detail (bool): Print all slide properties

        Returns:
            None
        """
        slide_path = Path(self.config.tcga_path).joinpath(f"wsi/{self.dataset}/")
        print(f"Dataset: {self.dataset.upper()}")
        print(f"Molecular data shape: {self.omic_df.shape}")
        sample_overlap = (set(self.omic_df["slide_id"]) & set(self.wsi_paths.keys()))
        print(f"Molecular/Slide match: {len(sample_overlap)}/{len(self.omic_df)}")
        # print(f"Slide dimensions: {slide.dimensions}")
        print(f"Slide level count: {self.sample_slide.level_count}")
        print(f"Slide level dimensions: {self.sample_slide.level_dimensions}")
        print(f"Slide resize dimensions: w: {self.wsi_width}, h: {self.wsi_height}")
        print(f"Sources selected: {self.sources}")
        print(f"Censored share: {np.round(len(self.omic_df[self.omic_df['censorship'] == 1])/len(self.omic_df), 3)}")
        print(f"Survival_bin_sizes: {dict(self.omic_df['y_disc'].value_counts().sort_values())}")

        if full_detail:
            pprint(dict(self.sample_slide.properties))

    def show_samples(self, n=1):
        """
        Logging util to show some detailed sample stats and render the whole slide image (e.g., in a notebook)
        Args:
            n (int): Number of samples to show

        Returns:
            None
        """
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




    def load_omic(self,
                  eps: float = 1e-6
                  ) -> pd.DataFrame:
        """
        Loads in omic data and returns a dataframe and filters depending on which whole slide images
        are available, such that only samples with both omic and WSI data are kept.
        Also calculates the discretised survival time for each sample.
        Args:
            eps (float): Epsilon value to add to min and max survival time to ensure all samples are included

        Returns:
            pd.DataFrame: Dataframe with omic data and discretised survival time (target)
        """
        data_path = Path(self.config.tcga_path).joinpath(f"omic/tcga_{self.dataset}_all_clean.csv.zip")
        df = pd.read_csv(data_path, compression="zip", header=0, index_col=0, low_memory=False)
        valid_subsets = ["all", "uncensored", "censored"]
        assert self.subset in valid_subsets, "Invalid cut specified. Must be one of 'all', 'uncensored', 'censored'"

        # handle missing values
        num_nans = df.isna().sum().sum()
        nan_counts = df.isna().sum()[df.isna().sum() > 0]
        df = df.fillna(df.mean(numeric_only=True))
        print(f"Filled {num_nans} missing values with mean")
        print(f"Missing values per feature: \n {nan_counts}")

        # filter samples for which there are no slides available
        if self.filter_overlap:
            slides_available = self.slide_ids
            omic_available = [id[:-4] for id in df["slide_id"]]
            overlap = set(slides_available) & set(omic_available)
            print(f"Slides available: {len(slides_available)}")
            print(f"Omic available: {len(omic_available)}")
            print(f"Overlap: {len(overlap)}")
            if len(slides_available) < len(omic_available):
                print(f"Filtering out {len(omic_available) - len(slides_available)} samples for which there are no omic data available")
                overlap_filter = [id + ".svs" for id in overlap]
                df = df[df["slide_id"].isin(overlap_filter)]
            elif len(slides_available) > len(omic_available):
                print(f"Filtering out {len(slides_available) - len(omic_available)} samples for which there are no slides available")
                self.slide_ids = overlap
            else:
                print("100% modality overlap, no samples filtered out")

        # assign target column (high vs. low risk in equal parts of survival)
        label_col = "survival_months"
        if self.subset == "all":
            df["y_disc"] = pd.qcut(df[label_col], q=self.n_bins, labels=False).values
        else:
            if self.subset == "censored":
                subset_df = df[df["censorship"] == 1]
            elif self.subset == "uncensored":
                subset_df = df[df["censorship"] == 0]
            # take q_bins from uncensored patients
            disc_labels, q_bins = pd.qcut(subset_df[label_col], q=self.n_bins, retbins=True, labels=False)
            q_bins[-1] = df[label_col].max() + eps
            q_bins[0] = df[label_col].min() - eps
            # use bin cuts to discretize all patients
            df["y_disc"] = pd.cut(df[label_col], bins=q_bins, retbins=False, labels=False, right=False, include_lowest=True).values

        df["y_disc"] = df["y_disc"].astype(int)

        if self.log_dir is not None:
            df.to_csv(self.log_dir.joinpath(f"{self.dataset}_omic_overlap.csv.zip"), compression="zip")

        return df

    def load_wsi(self, slide_id: str, level: int = None) -> Tuple:
        """
        Load in single slide and get region at specified resolution level
        Args:
            slide_id:
            level:
            resolution:

        Returns:
            Tuple (openslide object, tensor of region)
        """

        slide = OpenSlide(self.raw_path.joinpath(f"{slide_id}.svs"))

        # specify resolution level
        if level is None:
            level = slide.level_count # lowest resolution by default
        if level > slide.level_count - 1:
            level = slide.level_count - 1
        # load in region
        size = slide.level_dimensions[level]
        region = slide.read_region((0,0), level, size)
        # add transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :]), # remove alpha channel
            transforms.Resize((self.wsi_height, self.wsi_width)),
            RearrangeTransform("c h w -> h w c") # rearrange for Healnet architecture
        ])
        region_tensor = transform(region)
        return slide, region_tensor

    def load_patch_features(self, slide_id: str) -> torch.Tensor:
        """
        Loads patch features for a single slide from torch.pt file
        Args:
            slide_id (str): Slide ID

        Returns:
            torch.Tensor: Patch features
        """
        load_path = self.prep_path.joinpath(f"patch_features/{slide_id}.pt")
        with open(load_path, "rb") as file:
            patch_features = torch.load(file, weights_only=True)
        patch_features = patch_features.permute(1, 0)
        return patch_features



class SharedLRUCache:
    """
    Shared LRU cache for multiprocessing
    """
    def __init__(self, capacity: int):
        """

        Args:
            capacity (int): Number of items to be stored in the cache
        """
        manager = Manager()
        self.capacity = capacity
        self.cache = manager.dict()
        self.order = manager.list()
        self.lock = Lock()
    def get(self, key: int):
        with self.lock:
            if key in self.cache:
                # Move key to end to show it was recently used.
                self.order.remove(key)
                self.order.append(key)
                return self.cache[key]
            else:
                return None

    def set(self, key: int, value):
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
            else:
                if len(self.order) >= self.capacity:
                    removed_key = self.order.pop(0)  # Remove the first (least recently used) item.
                    del self.cache[removed_key]

            self.order.append(key)
            self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache


class RearrangeTransform(object):
    """
    Wrapper for einops.rearrange to pass into torchvision.transforms.Compose
    """
    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, img):
        img = rearrange(img, self.pattern)
        return img

class RepeatTransform(object):
    """
    Wrapper for einops.repeat to pass into torchvision.transforms.Compose
    """
    def __init__(self, pattern, b):
        self.pattern = pattern
        self.b = b
    def __call__(self, img):
        img = repeat(img, self.pattern, b=self.b)
        return img


if __name__ == '__main__':
    # os.chdir("../../")
    # config = Config("config/main.yml").read()
    # brca = TCGADataset("brca", config)
    # blca = TCGADataset("blca", config)
    # print(config)
    # print(brca.omic_df.shape, blca.omic_df.shape)
    # blca.load_wsi("TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F", resolution="lowest")

    from torch.utils.data import DataLoader
    
    n=50
    tab_tensor = torch.rand(size=(n, 1, 10))
    img_tensor = torch.rand(size=(n, 224, 224, 1))
    vid_tensor = torch.rand(size=(n, 12, 224, 224, 1))
    
    target = torch.rand(size=(n,))
    
    data = MMDataset([tab_tensor, img_tensor, vid_tensor], target)
    
    loader = DataLoader(data, batch_size=4, shuffle=True)
    
    # fetch batch
    tensors, target = next(iter(loader))
    
    print([t.shape for t in tensors])


# if __name__ == "__main__": 
    
