from pathlib import Path
from box import Box
import random
# from torch.utils.data import DataLoader
from healnet.etl import TCGADataset
from copy import deepcopy
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
from scipy.ndimage import zoom


class Explainer(object):
    def __init__(self, log_dir: str, show=False):
        self.log_dir = Path(log_dir)
        # run name
        self.show = show
        self.expl_dir = Path(f"explanations/{self.log_dir.name}")
        self.expl_dir.mkdir(parents=True, exist_ok=True)
        self.config = unpickle(self.log_dir.joinpath("config.pkl"))
        self.level = self.config["data.wsi_level"]
        self.dataset = self.config.dataset
        self.test_data_indices = unpickle(self.log_dir.joinpath("test_data_indices.pkl"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prep_path = Path(self.config["tcga_path"]).joinpath(f"wsi/{self.dataset}_preprocessed_level{self.level}/")
        self.raw_path = Path(self.config.tcga_path).joinpath(f"wsi/{self.dataset}")


        # plt.rcParams["font.family"] = "Avenir"

        print("Initializing dataset...")
        self.data = TCGADataset(
            dataset=self.dataset,
            config=self.config,
            level=self.level,
            sources=self.config.sources,
            n_bins=self.config["model_params.output_dims"],
            log_dir=None,
        )
        self.omic_df = self.data.omic_df
        print("Loading model...")
        self.model = self.load_model()
        self.model.eval()

    def run(self, n_high: int = 3, n_low: int = 0,
            downsample: float = None,
            run_omic: bool = True,
            run_slides: bool = True,
            heatmap: bool = True,
            highlight_patches: bool = True,
            save_patches: bool=True):
        """
        Run explanation for n_high high risk patients and n_low low risk patients
        Args:
            n_high:
            n_low:

        Returns:

        """
        self.high_risk = self.get_patients(risk="high", n=n_high)
        self.low_risk = self.get_patients(risk="low", n=n_low)
        self.heatmap = heatmap
        self.highlight_patches = highlight_patches

        # high risk
        # for i in range(n_high):
        for i in [2,3,5]:
        # for i in [2]: # final run
            self.save_name = f"high_risk_{i}"
            self.run_sample_explanation(self.high_risk[i:i+1], downsample=downsample, run_omic=run_omic, run_slides=run_slides, save_patches=save_patches)

        # low risk
        for i in range(n_low):
            self.run_sample_explanation(self.low_risk[i:i+1], downsample=downsample, run_omic=run_omic, run_slides=run_slides, save_patches=save_patches)



    def run_sample_explanation(self, sample: str, run_omic: bool = True, run_slides: bool = True, downsample: float = None, save_patches:bool=True):
        idx, slide_id = sample.index[0], sample.iloc[0]
        patch_coords = self.load_patch_coords(sample.iloc[0])
        self.slide, region = self.load_wsi(sample.iloc[0], level=self.level)

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
        omic_attn = [w for w in attn_weights if w.shape[2] == n_features]

        k = 20
        self.color = "Blues"
        self.pallete = sns.color_palette(self.color, n_colors=k)[::-1] # reverse order to get increasingly darker

        if len(omic_attn) > 0 and run_omic:
            # plot average
            self.plot_omic_attn(omic_attn, agg_layers=False, k=k)
            # plot_by layer
            # for i in range(len(omic_attn)):
            #     self.plot_omic_attn(omic_attn, layer=i, agg_layers=False)

        if len(slide_attn) > 0 and run_slides:
            print(f"Reading slide...")

            slide_img = self.slide.read_region(location=(0, 0), level=self.level, size=self.slide.level_dimensions[self.level])

            # save original image
            plt.imshow(slide_img)
            plt.axis('off')
            save_path = self.expl_dir.joinpath(f"{self.save_name}_original.png")
            slide_img.save(save_path)
            # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.show()

            slide_img = np.array(slide_img)[:, :, :3]  # Convert PIL image to array and remove alpha if present

            # pick layer with highest std deviation across patches
            layer_to_viz = np.argmax([torch.std(w).detach().cpu() for w in slide_attn])
            # None --> mean across all layers
            self.plot_slide_attn(slide_img, slide_attn, patch_coords, layer=None, downsample=downsample, save_patches=save_patches)




            # for layer in range(len(slide_attn)):
            #     try:
            #         self.plot_slide_attn(slide_img, slide_attn, patch_coords, layer=layer)
            #     except:
            #         pass


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
            layer_to_viz = np.argmax([torch.std(w).detach().cpu() for w in omic_attn])
            omic_attn = torch.mean(omic_attn[layer_to_viz], dim=1).detach().cpu().numpy()
        feats = self.data.features.columns.tolist()
        plot_df = pd.DataFrame({"feature": feats, "attention": omic_attn.squeeze()}).sort_values(by="attention", ascending=False)
        # filter "age" and "is_female" (not omic feature)
        plot_df = plot_df[~plot_df["feature"].str.contains("age|is_female")]
        # take top scale_fraction features
        # plot_df = plot_df.iloc[:int(scale_fraction * len(feats))]


        min_attn = plot_df["attention"].min()
        max_attn = plot_df["attention"].max()
        plot_df = plot_df.iloc[:k]
        # min_attn = np.percentile(plot_df["attention"], 40)
        plot_df["attention_scaled"] = ((plot_df["attention"] - min_attn) / (max_attn - min_attn) / k)
        # apply sigmoid
        plt.figure(figsize=(6, 10))
        sns.barplot(data=plot_df, y="feature", x="attention_scaled", palette=self.pallete, )
        lower_lim = plot_df["attention_scaled"].min() - 0.005 * plot_df["attention_scaled"].min()
        uppler_lim = plot_df["attention_scaled"].max() + 0.001 * plot_df["attention_scaled"].max()
        # lower_lim =
        # uppler_lim = plot
        plt.xlim(lower_lim, uppler_lim)
        # if agg_layers:
        #     # plt.title(f"Mean Omic Attention")
        # else:
        #     plt.title(f"Layer {layer+1} Omic Attention")
        plt.yticks(fontsize=12, rotation=30)
        plt.xticks(fontsize=12, rotation=30)
        plt.xlabel("Attention Scaled", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.subplots_adjust(left=0.3)
        save_path = self.expl_dir.joinpath(f"{self.save_name}_omic_attn.png")
        print(f"Saving to {save_path}")
        plt.savefig(self.expl_dir.joinpath(f"{self.save_name}_omic_attn.png"))
        if self.show:
            plt.show()

    def plot_slide_attn(self, slide_img: np.array,
                        slide_attn: torch.Tensor,
                        patch_coords: np.array,
                        layer: int=0, downsample: float = None, save_patches: bool=True):

        patch_size = (256, 256)
        if layer is None:
            # take mean across all layers
            slide_attn = torch.stack(slide_attn).mean(dim=0).mean(dim=1).squeeze().detach().cpu().numpy()
        else:
            slide_attn = torch.mean(slide_attn[layer], dim=1).squeeze().detach().cpu().numpy()
        slide_attn = slide_attn[:len(patch_coords)]
        x_coord = patch_coords[:, 0]
        y_coord = patch_coords[:, 1]
        plot_df = pd.DataFrame({"x": x_coord, "y": y_coord, "attention": slide_attn})
        # plot_df = pd.DataFrame({"patch": patch_coords, "attention": slide_attn}).sort_values(by="attention", ascending=False)
        # normalize attention scores
        slide_dims = self.slide.level_dimensions
        original_dims  = slide_dims[0]
        # scale X depending on which level we are reading the slide at
        scale_factor = int(original_dims[0] / slide_dims[self.level][0])
        plot_df["x_scaled"] = (plot_df["x"] / scale_factor).astype(int)
        plot_df["y_scaled"] = (plot_df["y"] / scale_factor).astype(int)
        plot_df["attention_scaled"] = (plot_df["attention"] - plot_df["attention"].min()) / (plot_df["attention"].max() - plot_df["attention"].min())

        if downsample is not None:
            slide_img = zoom(np.array(slide_img), (downsample, downsample, 1))
            plot_df['x_scaled'] = (plot_df['x_scaled'] * downsample).astype(int)
            plot_df['y_scaled'] = (plot_df['y_scaled'] * downsample).astype(int)
            patch_size = (int(patch_size[0] * downsample), int(patch_size[1] * downsample))

        if self.highlight_patches:
            self.highlight_top_patches(slide_img=slide_img, df=plot_df, patch_size=patch_size, show=True, layer=layer)
        if self.heatmap:
            self.create_heatmap(slide_img=slide_img, patch_size=patch_size, df=plot_df, show=True, layer=layer)


        # save top 5 patches as images
        if save_patches:
            top_df = plot_df.sort_values(by='attention_scaled', ascending=False).head(5)
            for index, row in top_df.iterrows():
                x, y = int(row['x_scaled']), int(row['y_scaled'])
                patch = slide_img[y:y+patch_size[1], x:x+patch_size[0]]
                plt.imshow(patch)
                plt.axis('off')
                save_path = self.expl_dir.joinpath(f"{self.save_name}_patch_{index}.png")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                print(f"Saving to {save_path}")
                plt.show()

            # also save in highest res
            for index, row in top_df.iterrows():
                x, y = int(row["x"]), int(row["y"])
                patch_size_orig = (256 * scale_factor, 256*scale_factor)
                print(patch_size_orig)
                patch = self.slide.read_region(location=(x, y), level=0, size=patch_size_orig)
                save_path = self.expl_dir.joinpath(f"{self.save_name}_patch_{index}_high_res.png")
                patch.save(save_path)
                plt.imshow(patch)
                plt.axis("off")
                plt.show()


    def highlight_top_patches(self, slide_img, df, patch_size, show=True, layer: int=1):


        """
        Highlights the top 5 features with highest attention scores on an OpenSlide image using a frame.

        Parameters:
        - slide_img: PIL image of the entire slide read in at specified level
        - df: DataFrame containing 'x_scaled', 'y_scaled', and 'attention_scaled' columns.
        - patch_size: size of each patch to frame.
        - show: Boolean to decide whether to display the image with highlighted features.

        Returns:
        - Image with highlighted top features.
        """
        print(f"Highlighting top patches...")

        # Sort dataframe by attention_scaled in descending order and take top 5
        top_df = df.sort_values(by='attention_scaled', ascending=False).head(5)

        # Initialize a figure for plotting
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(slide_img)

        # Draw frames for the top 5 features
        for index, row in top_df.iterrows():
            x, y = int(row['x_scaled']), int(row['y_scaled'])
            rect = patches.Rectangle((x, y), patch_size[0], patch_size[1], linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

        # Hide axes and display image if show is True
        plt.axis('off')
        plt.savefig(self.expl_dir.joinpath(f"{self.save_name}_patch_highlights.png"), dpi=300)
        if self.show:
            # plt.title(f"Layer {layer+1} High Attention Patches", size=15)
            plt.show()



    def create_heatmap(self, slide_img, df, patch_size, show=True, layer: int=1, mask_cutoff: float=0.0):
        """
        Creates a heatmap from attention scores on an OpenSlide image.

        Parameters:
        - slide_img: PIL image of the entire slide read in at specified level
        - df: DataFrame containing 'x', 'y', and 'attention' columns.
        - patch_size: size of each patch to highlight.
        - colormap: colormap to use for heatmap.

        Returns:
        - heatmap overlayed on the slide image.
        """

        print(f"Creating heatmap...")
        # Create an empty heatmap of same size as slide
        heatmap = np.zeros(slide_img.shape[:2])

        # Convert DataFrame columns to NumPy arrays
        xs = df['x_scaled'].values.astype(int)
        ys = df['y_scaled'].values.astype(int)
        attentions = df['attention_scaled'].values
        for x, y, attention in zip(xs, ys, attentions):
            heatmap[y:y+patch_size[1], x:x+patch_size[0]] = attention
        mask = heatmap <= mask_cutoff

        # save heatmap and mask
        np.save(self.expl_dir.joinpath(f"{self.save_name}_heatmap.npy"), heatmap)
        np.save(self.expl_dir.joinpath(f"{self.save_name}_mask.npy"), mask)

        # Show the heatmap and legend if the 'show' parameter is set to True
        if show:
            plt.figure(figsize=(10, 10))  # You can adjust the figure size as needed

            plt.imshow(slide_img)
            cbar_kws = {"shrink": 0.5} # colour bar scale
            ax = sns.heatmap(heatmap, cmap=self.color, alpha=0.7,
                             cbar=True, annot=False, cbar_kws=cbar_kws, mask=mask)
            # Add colorbar to represent the attention values
            cbar = ax.collections[0].colorbar
            cbar.set_ticks([0, heatmap.max()])
            cbar.set_label('Attention', size=15)
            # plt.title(f"Layer {layer+1} Attention Heatmap", size=15)
            plt.axis('off')
            plt.savefig(self.expl_dir.joinpath(f"{self.save_name}_heatmap.png"), dpi=300)
            if self.show:
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
    # log_path = "logs/devoted-universe-2701" # ucec level 1, omic attentio
    # log_path = "logs/pleasant-smoke-2704" # kirp level 2 (dev), omic attention
    # log_path= "logs/rural-sky-2709" # kirp level 1, omic attention
    # log_path = "logs/glad-meadow-3551" # ucec level 2, omic attention
    log_path="logs/smart-sweep-35" # ucec level 1, omic attention


    torch.multiprocessing.set_start_method("fork")

    e = Explainer(log_path, show=True)
    # e.run(n_high=3, n_low=0, downsample=None)
    e.run(n_high=10,
          n_low=0,
          downsample=0.25,
          run_omic=False,
          run_slides=True,
          heatmap=True,
          highlight_patches=False,
          save_patches=True,
          )

    # e.run(e.high_risk[2:3])

    #
    # for id in e.high_risk:
    #     e.run(id)
    # for id in e.low_risk:
    #     e.run(id)


