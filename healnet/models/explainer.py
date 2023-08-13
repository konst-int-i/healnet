from pathlib import Path
from box import Box
# from torch.utils.data import DataLoader
from healnet.etl import TCGADataset
from healnet.models import HealNet
from healnet.utils import unpickle
import seaborn as sns
import numpy as np
import torch
from typing import *
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
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
        self.slide, region = self.load_wsi(sample.iloc[0], level=self.level)

        # plot slide in matplotlib
        # plt.imshow(region)
        # plt.show()

        (omic_tensor, slide_tensor), _, _, _ = self.data[idx]

        slide_tensor = slide_tensor.unsqueeze(0).to(self.device)
        n_patches = slide_tensor.shape[1]
        # add batch dim
        omic_tensor = omic_tensor.unsqueeze(0).to(self.device)
        n_features= omic_tensor.shape[1]
        logits = self.model([omic_tensor, slide_tensor])
        probs = torch.softmax(logits, dim=1)
        attn_weights = self.model.get_attention_weights()
        slide_attn = [w for w in attn_weights if w.shape[2] == n_patches]
        # slide_attn = [w for w in attn_weights if w.shape[1] == n_patches]


        # omic_attn = [w for w in attn_weights if w.shape[2] == n_features]

        # if len(omic_attn) > 0:
        #     # plot average
        #     self.plot_omic_attn(omic_attn, agg_layers=True)
        #     # plot_by layer
        #     for i in range(len(omic_attn)):
        #         self.plot_omic_attn(omic_attn, layer=i, agg_layers=False)

        if len(slide_attn) > 0:
            # pick layer with highest std deviation across patches
            layer_to_viz = np.argmax([torch.std(w).detach().cpu() for w in slide_attn])

            self.plot_slide_attn(slide_attn, patch_coords, layer=layer_to_viz)


    def plot_omic_attn(self, omic_attn, k: int=20, scale_fraction: float=0.5, layer: int=0, agg_layers: bool=False):
        """

        Args:
            omic_attn:
            k:
            scale_fraction:
            layer:

        Returns:

        """
        if agg_layers:
            # average attention across 1) all layers and 2) all heads
            omic_attn = torch.stack(omic_attn).mean(dim=0).mean(dim=1).detach().cpu().numpy()
        else:
        # only first layer
            omic_attn = torch.mean(omic_attn[layer], dim=1).detach().cpu().numpy()
        feats = self.data.features.columns.tolist()
        plot_df = pd.DataFrame({"feature": feats, "attention": omic_attn.squeeze()}).sort_values(by="attention", ascending=False)
        # take top scale_fraction features
        plot_df = plot_df.iloc[:int(scale_fraction * len(feats))]


        min_attn = plot_df["attention"].min()
        max_attn = plot_df["attention"].max()
        plot_df["attention_scaled"] = (plot_df["attention"] - min_attn) / (max_attn - min_attn)
        plot_df = plot_df.iloc[:k]
        sns.barplot(data=plot_df, y="feature", x="attention_scaled")
        lower_lim = plot_df["attention_scaled"].min() - 0.05
        plt.xlim(lower_lim, 1)
        if agg_layers:
            plt.title(f"Mean Omic Attention")
        else:
            plt.title(f"Layer {layer+1} Omic Attention")
        plt.yticks(fontsize=8)
        plt.subplots_adjust(left=0.3)
        plt.show()

    def plot_slide_attn(self, slide_attn, patch_coords, layer: int=0):
        slide_attn = torch.mean(slide_attn[layer], dim=1).squeeze().detach().cpu().numpy()
        slide_attn = slide_attn[:len(patch_coords)]
        x_coord = patch_coords[:, 0]
        y_coord = patch_coords[:, 1]
        plot_df = pd.DataFrame({"x": x_coord, "y": y_coord, "attention": slide_attn})
        # plot_df = pd.DataFrame({"patch": patch_coords, "attention": slide_attn}).sort_values(by="attention", ascending=False)
        # normalize attention scores
        slide_dims = self.slide.level_dimensions
        original_dims  = slide_dims[0]
        scale_factor = int(original_dims[0] / slide_dims[self.level][0])
        plot_df["x_scaled"] = (plot_df["x"] / scale_factor).astype(int)
        plot_df["y_scaled"] = (plot_df["y"] / scale_factor).astype(int)
        plot_df["attention_scaled"] = (plot_df["attention"] - plot_df["attention"].min()) / (plot_df["attention"].max() - plot_df["attention"].min())
        self.create_heatmap(slide=self.slide, df=plot_df)


    def create_heatmap(self, slide, df, patch_size=(256, 256), show=True, color="red"):
        """
        Creates a heatmap from attention scores on an OpenSlide image.

        Parameters:
        - slide_path: path to the OpenSlide image.
        - df: DataFrame containing 'x', 'y', and 'attention' columns.
        - patch_size: size of each patch to highlight.
        - colormap: colormap to use for heatmap.

        Returns:
        - heatmap overlayed on the slide image.
        """

        # Open the slide
        # slide = openslide.OpenSlide(slide_path)

        # Read the entire slide
        slide_img = slide.read_region(location=(0, 0), level=self.level, size=slide.level_dimensions[self.level])
        slide_img = np.array(slide_img)[:, :, :3]  # Convert PIL image to array and remove alpha if present

        # Create an empty heatmap of same size as slide
        heatmap = np.zeros(slide_img.shape[:2])

        # Populate heatmap using the attention scores from the DataFrame
        for index, row in df.iterrows():
            x, y = int(row['x_scaled']), int(row['y_scaled'])
            attention = row['attention_scaled']
            heatmap[y:y+patch_size[1], x:x+patch_size[0]] = attention


        # Create a heatmap using matplotlib's colormaps
        # heatmap_colored = (color_map[:, :, :3] * heatmap[:, :, np.newaxis] * 255).astype(np.uint8)

        # Create a mask for blending
        # mask = (heatmap > 0).astype(np.uint8)[:, :, np.newaxis]
        # blended = slide_img * (1 - mask) + heatmap * mask

        # Show the heatmap and legend if the 'show' parameter is set to True
        if show:
            plt.figure(figsize=(10, 10))  # You can adjust the figure size as needed

            plt.imshow(slide_img)
            cbar_kws = {"shrink": 0.5} # colour bar scale
            ax = sns.heatmap(heatmap, cmap=sns.light_palette(color, as_cmap=True), alpha=0.3,
                             cbar=True, annot=False, cbar_kws=cbar_kws)

            # Add colorbar to represent the attention values
            cbar = ax.collections[0].colorbar
            cbar.set_ticks([0, heatmap.max()])
            cbar.set_label('Attention', size=15)

            plt.axis('off')
            plt.show()


        # return blended





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
    # log_path = "logs/pleasant-smoke-2704" # kirp level 2 (dev), omic attention
    log_path = "logs/solar-paper-2708" # ucec level 2, no omic attention

    torch.multiprocessing.set_start_method("fork")

    e = Explainer(log_path)
    e.run(e.high_risk[2:3])


