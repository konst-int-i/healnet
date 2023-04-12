from invoke import task
from x_perceiver.utils import Config
from pathlib import Path
import pandas as pd
import os

@task
def download(c, dataset, samples: int = None, config_path="config/main.yml"):
    valid_datasets = ["brca", "blca"]
    config = Config(config_path).read()
    download_dir = Path(config.tcga_path).joinpath(f"wsi/{dataset}")

    assert dataset in valid_datasets, f"Invalid dataset arg, must be one of {valid_datasets}"

    manifest_path = Path(config.tcga_path).joinpath(f"gdc_manifests/filtered/{dataset}_wsi_manifest_filtered.txt")
    # Download entire manifest unless specified otherwise
    if samples is not None:
        manifest = pd.read_csv(manifest_path, sep="\t")
        manifest = manifest.sample(n=int(samples), random_state=42)
        tmp_path = manifest_path.parent.joinpath(f"{dataset}_tmp.txt")
        manifest.to_csv(tmp_path, sep="\t", index=False)
        print(f"Downloading {manifest.shape[0]} files from {dataset} dataset...")
        c.run(f"{config.gdc_client} download -m {tmp_path} -d {download_dir}")
        # cleanup
        os.remove(tmp_path)

    else:
        c.run(f"{config.gdc_client} download -m {manifest_path} -d {download_dir}")