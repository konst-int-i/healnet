from healnet.etl import TCGADataset
from healnet.utils import Config
from pathlib import Path

def filter_manifest_files(config: Config, dataset: str):
    """
    Temporary util to filter the manifest files to only include the
    WSI images required
    Returns:
    """
    mol_df = TCGADataset(dataset, config).omic_df
    manifest_path = Path(config.tcga_path).joinpath(f"gdc_manifests/full/{dataset}_wsi_manifest_full.txt")
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    manifest_filtered = manifest_df.loc[manifest_df.filename.isin(mol_df["slide_id"])]

    assert manifest_filtered.shape[0] == mol_df.shape[0], "Number of filtered manifest files does not match number of omic files"

    write_path = Path(config.tcga_path).joinpath(f"gdc_manifests/filtered/{dataset}_wsi_manifest_filtered.txt")
    manifest_filtered.to_csv(write_path, sep="\t", index=False)
    print(f"Saved filtered manifest file to {write_path}")
    return None


