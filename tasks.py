from invoke import task
from x_perceiver.utils import Config
from pathlib import Path
import pandas as pd
import os


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
def download(c, dataset:str, config:str, samples: int = None):
    valid_datasets = ["brca", "blca"]
    conf = Config(config).read()
    download_dir = Path(conf.tcga_path).joinpath(f"wsi/{dataset}")

    assert dataset in valid_datasets, f"Invalid dataset arg, must be one of {valid_datasets}"

    manifest_path = Path(conf.tcga_path).joinpath(f"gdc_manifests/filtered/{dataset}_wsi_manifest_filtered.txt")
    # Download entire manifest unless specified otherwise
    if samples is not None:
        manifest = pd.read_csv(manifest_path, sep="\t")
        manifest = manifest.sample(n=int(samples), random_state=42)
        tmp_path = manifest_path.parent.joinpath(f"{dataset}_tmp.txt")
        manifest.to_csv(tmp_path, sep="\t", index=False)
        print(f"Downloading {manifest.shape[0]} files from {dataset} dataset...")
        c.run(f"{conf.gdc_client} download -m {tmp_path} -d {download_dir}")
        # cleanupJ
        os.remove(tmp_path)

    else:
        c.run(f"{conf.gdc_client} download -m {manifest_path} -d {download_dir}")

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
def preprocess(c, dataset: str, config: str, level: int):
    """
    Preprocesses WSI images for downstream tasks.
    Args:
        c:
        dataset:
        config:

    Returns:

    """
    conf = Config(config).read()
    raw_dir = Path(conf.tcga_path).joinpath(f"wsi/{dataset}")
    preprocessed_dir = Path(conf.tcga_path).joinpath(f"wsi/{dataset}_preprocessed_level{level}")

    # clone CLAM repo if doesn't exist
    if not os.path.exists("CLAM/"):
        c.run("git clone git@github.com:mahmoodlab/CLAM.git")

    c.run(f"python CLAM/create_patches_fp.py --source {raw_dir} --save_dir {preprocessed_dir} "
          f"--patch_size {config['data.patch_size']} --patch_level {level} --seg --patch --stitch")

