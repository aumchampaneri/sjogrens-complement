# Download Salivary gland data from cxg
#
#
# %%
from pathlib import Path

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "outputs/salivary/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_DIR / "data/salivary/"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% IMPORTS
import cellxgene_census

# %%
census = cellxgene_census.open_soma(census_version="latest")
census["census_info"]["summary"].read().concat().to_pandas()
datasets = census["census_info"]["datasets"].read().concat().to_pandas()

# Fetch an AnnData object
dataset_id = "df1edb87-e512-43ae-b5f4-cb179cfc2bb4"  # https://datasets.cellxgene.cziscience.com/31380664-ba9c-49d1-9961-b2bf4f7131a2.h5ad
cellxgene_census.download_source_h5ad(
    dataset_id, to_path=f"{DATA_DIR}/salivary-cxg.h5ad", progress_bar=True
)
census.close()

# %%
# Get a citation string for the slice
datasets[datasets["dataset_id"] == dataset_id].iloc[0]
