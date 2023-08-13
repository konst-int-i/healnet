from pathlib import Path
from box import Box
# from torch.utils.data import DataLoader
from healnet.etl import TCGADataset
from healnet.models import HealNet
from healnet.utils import unpickle
import torch
from typing import *
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from openslide import OpenSlide

class Explainer(object):
    def __init__(self, log_dir: str, level: int = 2):
        self.log_dir = Path(log_dir)
        self.config = unpickle(self.log_dir.joinpath("config.pkl"))
        self.level = self.config["data.wsi_level"]
        self.dataset = self.config.dataset
        self.test_data_indices = unpickle(self.log_dir.joinpath("test_data_indices.pkl"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prep_path = Path(self.config["tcga_path"]).joinpath(f"wsi/{self.dataset}_preprocessed_level{self.level}/")
        self.raw_path = Path(self.config.tcga_path).joinpath(f"wsi/{self.dataset}")

        # self.omic_df = self.load_omic_df()

        self.data = TCGADataset(
            dataset=self.dataset,
            config=self.config,
            level=self.level,
            sources=self.config.sources,
            n_bins=self.config["model_params.output_dims"],
            log_dir=None,
        )
        self.omic_df = self.data.omic_df
        self.model = self.load_model()

        self.high_risk = self.get_patients(risk="high")
        self.low_risk = self.get_patients(risk="low")

    def run(self, sample):
        idx, slide_id = sample.index[0], sample.iloc[0]
        patch_coords = self.load_patch_coords(sample.iloc[0])
        slide, region = self.load_wsi(sample.iloc[0], level=self.level)

        # plot slide in matplotlib
        # region.show()
        plt.imshow(region)
        plt.show()

        (omic_tensor, slide_tensor), _, _, _ = self.data[idx]
        # add batch dim
        omic_tensor = omic_tensor.unsqueeze(0).to(self.device)
        slide_tensor = slide_tensor.unsqueeze(0).to(self.device)
        logits = self.model([omic_tensor, slide_tensor])
        probs = torch.softmax(logits, dim=1)
        attn_weights = self.model.get_attention_weights()
        slide_weights = [w for w in attn_weights if w.shape[2] > 1]

        layer = 0
        slide_weights[0]

        print(attn_weights)


    def load_model(self):
        state_dict = torch.load(self.log_dir.joinpath("best_model.pt"), map_location=self.device)

        feat, _, _, _ = next(iter(self.data))

        num_sources = len(self.config["sources"])
        if num_sources == 1:
            input_channels = [feat[0].shape[1]]
            input_axes = [1]
            modalities = 1
        elif num_sources == 2:
            input_channels = [feat[0].shape[1], feat[1].shape[1]]
            input_axes = [1, 1]
            modalities = 2

        # reload model
        model = HealNet(
                modalities=modalities,
                input_channels=input_channels, # number of features as input channels
                input_axes=input_axes, # second axis (b n_feats c)
                num_classes=self.config[f"model_params.output_dims"],
                num_freq_bands=self.config[f"model_params.num_freq_bands"],
                depth=self.config[f"model_params.depth"],
                max_freq=self.config[f"model_params.max_freq"],
                num_latents = self.config[f"model_params.num_latents"],
                latent_dim = self.config[f"model_params.latent_dim"],
                cross_dim_head = self.config[f"model_params.cross_dim_head"],
                latent_dim_head = self.config[f"model_params.latent_dim_head"],
                cross_heads = self.config[f"model_params.cross_heads"],
                latent_heads = self.config[f"model_params.latent_heads"],
                attn_dropout = self.config[f"model_params.attn_dropout"],
                ff_dropout = self.config[f"model_params.ff_dropout"],
                weight_tie_layers = self.config[f"model_params.weight_tie_layers"],
                fourier_encode_data = self.config[f"model_params.fourier_encode_data"],
                self_per_cross_attn = self.config[f"model_params.self_per_cross_attn"],
                final_classifier_head = True,
                snn = self.config[f"model_params.snn"],
            )
        model.float()
        model.to(self.device)

        # recover state
        model.load_state_dict(state_dict)

        return model


    def load_patch_coords(self, slide_id: str):
        patch_path = self.prep_path.joinpath(f"patches/{slide_id}.h5")
        h5_file = h5py.File(patch_path, "r")
        patch_coords = h5_file["coords"][:]

        return patch_coords

    # def load_slide(self, slide_id: str):

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

        # load in openslide object
        # slide_path = self.wsi_paths[slide_id]
        # slide = OpenSlide(slide_path + ".svs")
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
        return slide, region




    def load_omic_df(self):

        data_path = Path(self.log_dir).joinpath(f"{self.dataset}_omic_overlap.csv.zip")
        df = pd.read_csv(data_path, compression="zip").drop("Unnamed: 0", axis=1)

        return df

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

        filtered = filtered.sort_values(by=["y_disc", "survival_months"], ascending=ascending)

        risk_ids = filtered.iloc[:n]["slide_id"]
        # filter file extensions
        risk_ids = risk_ids.apply(lambda x: x[:-4])

        return risk_ids



if __name__ == "__main__":
    # log_path = "logs/blca_09-08-2023_17-36-36"
    # log_path = "logs/kirp_10-08-2023_18-31-30"
    # log_path = "logs/graceful-haze-2166"
    # log_path= "logs/celestial-blaze-2702" # kirp level 1 (final)
    log_path = "logs/polar-firebrand-2703" # kirp level 2 (dev), no omic attention

    torch.multiprocessing.set_start_method("fork")

    e = Explainer(log_path)
    e.run(e.high_risk[2:3])


