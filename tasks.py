from invoke import task
from x_perceiver.utils import Config
from torchvision import transforms
from pathlib import Path
from openslide import OpenSlide
import pandas as pd
import torch
from tqdm import tqdm
import os
import h5py
import torchvision.models as models


@task
def install(c, system: str):
    assert system in ["linux", "mac"], "Invalid OS specified, must be one of 'linux' or 'mac'"

    print(f"Installing gdc-client for {system}...")
    if system == "linux":
        c.run("curl -0 https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip "
              "--output gdc-client.zip")
        c.run("unzip gdc-client.zip")
    if system == "mac":
        c.run("curl -0 https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_OSX_x64.zip "
              "--output gdc-client.zip")
        c.run("unzip gdc-client.zip")
    print(f"Installed gdc-client at {os.getcwd()}")
    # cleanup
    os.remove("gdc-client.zip")

@task
def download(c, dataset:str, config:str="config/main_gpu.yml", samples: int = None):
    valid_datasets = ["brca", "blca", "kirp", "ucec", "hnsc", "paad", "luad", "lusc"]
    conf = Config(config).read()
    download_dir = Path(conf.tcga_path).joinpath(f"wsi/{dataset}")

    # create download dir if doesn't exist (first time running)
    if not download_dir.exists():
        download_dir.mkdir(parents=True)

    assert dataset in valid_datasets, f"Invalid dataset arg, must be one of {valid_datasets}"

    manifest_path = Path(f"./data/tcga/gdc_manifests/filtered/{dataset}_wsi_manifest_filtered.txt")
    # manifest_path = Path(conf.tcga_path).joinpath(f"gdc_manifests/filtered/{dataset}_wsi_manifest_filtered.txt")
    # Download entire manifest unless specified otherwise
    if samples is not None:
        manifest = pd.read_csv(manifest_path, sep="\t")
        manifest = manifest.sample(n=int(samples), random_state=42)
        tmp_path = manifest_path.parent.joinpath(f"{dataset}_tmp.txt")
        manifest.to_csv(tmp_path, sep="\t", index=False)
        print(f"Downloading {manifest.shape[0]} files from {dataset} dataset...")
        c.run(f"{conf.gdc_client} download -m {tmp_path} -d {download_dir}")
        # cleanup
        os.remove(tmp_path)

    else:
        command = f"{conf.gdc_client} download -m {manifest_path} -d {download_dir}"
        try:
            c.run(command)
        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"Command: {command}")

    # flatten directory structure (required to easily run CLAM preprocessing)
    flatten(c, dataset, config)

@task
def flatten(c, dataset: str, config: str):
    """
    Flattens directory structure for WSI images after download using the GDC client from
     `data_dir/*.svs` instead of `data_dir/hash_subdir/*.svs`.
    Args:
        c:
        dataset:
        config:

    Returns:
    """
    conf = Config(config).read()
    download_dir = Path(conf.tcga_path).joinpath(f"wsi/{dataset}")
    # flatten directory structure
    c.run(f"find {download_dir} -type f -name '*.svs' -exec mv {{}} {download_dir} \;")
    # remove everything that's not a .svs file
    c.run(f"find {download_dir} ! -name '*.svs' -delete")

@task
def preprocess(c, dataset: str, level: int, config: str="config/main_gpu.yml", step:str= "patch"):
    """
    Preprocesses WSI images for downstream tasks.
    Args:
        c:
        dataset:
        config:

    Returns:

    """
    conf = Config(config).read()
    raw_path = Path(conf.tcga_path).joinpath(f"wsi/{dataset}")
    prep_path = Path(conf.tcga_path).joinpath(f"wsi/{dataset}_preprocessed_level{level}")

    valid_steps = ["patch", "features"]
    assert step in valid_steps, f"Invalid step arg, must be one of {valid_steps}"

    # clone CLAM repo if doesn't exist
    if not os.path.exists("CLAM/"):
        c.run("git clone git@github.com:mahmoodlab/CLAM.git")

    if step == "patch":
        c.run(f"python CLAM/create_patches_fp.py --source {raw_path} --save_dir {prep_path} "
          f"--patch_size {int(conf.data.patch_size)} --patch_level {int(level)} --seg --patch --stitch")

    if step == "features":
        slide_ids = [x.rstrip(".svs") for x in os.listdir(raw_path)]
        # load patch coords
        coords = {}
        for slide_id in slide_ids:
            patch_path = prep_path.joinpath(f"patches/{slide_id}.h5")
            try:
                h5_file = h5py.File(patch_path, "r")
                patch_coords = h5_file["coords"][:]
                coords[slide_id] = patch_coords
            except FileNotFoundError as e:
                print(f"No patches available for file {patch_path}")
                pass
        max_patches = max([coords.get(key).shape[0] for key in coords.keys()])
        print(f"Max patches: {max_patches}")

        # load in resnet50 model
        # patch_encoder = models.resnet50(pretrained=True)
        patch_encoder = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V2)
        patch_encoder = torch.nn.Sequential(*(list(patch_encoder.children())[:-1])) # remove classifier head
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        patch_encoder.to(device)
        patch_encoder.eval()

        patch_tensors = torch.zeros(max_patches, 2048)
        num_slides = len(slide_ids)
        # extract features
        for slide_count, slide_id in enumerate(coords.keys()):
            slide = OpenSlide(raw_path.joinpath(f"{slide_id}.svs"))
            print(f"slide {slide_count+1}/{num_slides}")

            for idx, coord in enumerate(tqdm(coords[slide_id])):
                # print(f"{dataset.upper()} Level {level}: Processing patch {idx} of {len(coords[slide_id])} for "
                #       f"slide {slide_count+1}/{num_slides}")
                x, y = coord
                region_transform = transforms.Compose([
                transforms.Lambda(lambda image: image.convert("RGB")), # need to convert to RGB for ResNet encoding
                transforms.ToTensor(),
                transforms.Resize((224, 224)), # resize in line with ResNet50
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalisation
                ])
                patch_region = region_transform(slide.read_region((x, y), level=int(level), size=(256, 256)))
                patch_region = patch_region.to(device)
                patch_region = patch_region.unsqueeze(0)
                patch_features = patch_encoder(patch_region)
                patch_tensors[idx] = patch_features.cpu().detach().squeeze()

            # save features
            feat_path = prep_path.joinpath("patch_features")
            if not feat_path.exists():
                feat_path.mkdir(parents=False)
            save_path = feat_path.joinpath(f"{slide_id}.pt")
            torch.save(patch_tensors, save_path)


