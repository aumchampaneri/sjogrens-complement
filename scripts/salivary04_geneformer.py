# Geneformer
#
#
# %%
import os
from pathlib import Path

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "outputs" / "salivary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_DIR / "data" / "salivary"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIR = DATA_DIR / "geneformer" / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZED_DIR = DATA_DIR / "geneformer" / "tokenized"
TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

RESOURCE_DIR = PROJECT_DIR / "resources"

GENEFORMER = PROJECT_DIR / "geneformer"
GENEFORMER.mkdir(parents=True, exist_ok=True)

# %% IMPORTS
import subprocess

import scanpy as sc

import geneformer

# %% DOWNLOAD GENEFORMER
if not os.listdir(GENEFORMER):
    print("Initializing Git LFS and cloning Geneformer from Hugging Face...")

    cmds = [
        "git lfs install",
        f"git clone https://huggingface.co/ctheodoris/Geneformer {GENEFORMER}",
    ]

    for cmd in cmds:
        result = subprocess.run(cmd, shell=True, check=True, text=True)
    print("Clone complete!")
else:
    print(f"Geneformer directory at {GENEFORMER} is not empty. Skipping clone.")

# %%
adata = sc.read_h5ad(DATA_DIR / "salivary-cxg.h5ad")

# %%
# Duplicate index to a new .var column -> "ensembl_id"
adata.var["ensembl_id"] = adata.var.index

# Sjogren syndrome
# normal (control)
set(adata.obs["disease"])

# cell type annotations from original authors
set(adata.obs["cell_type"])

# collapse macrophage and plasma cell subtypes into broader categories
rename_dict = {
    "alternatively activated macrophage": "Macrophage",
    "inflammatory macrophage": "Macrophage",
    "IgA plasma cell": "Plasma cell",
    "IgG plasma cell": "Plasma cell",
}

# create unified cell-type annotation
adata.obs["cell_type_union"] = [
    rename_dict.get(ct, ct) for ct in adata.obs["cell_type"]
]
set(adata.obs["cell_type_union"])

# reformat developmental stage / age attribute
set(adata.obs["development_stage"])

# keep only first two characters (e.g., "Adult" -> "Ad")
adata.obs["development_stage"] = [ds[:2] for ds in adata.obs["development_stage"]]
set(adata.obs["development_stage"])

# select cell types relevant to complement production, regulation, and signaling
included_cell_types = [
    "Macrophage",
    "dendritic cell",  # ← lowercase d
    "B cell",
    "Plasma cell",
    "CD4-positive, alpha-beta T cell",
    "CD8-positive, alpha-beta cytotoxic T cell",
    "CD8-positive, alpha-beta regulatory T cell",  # ← new
    "effector CD8-positive, alpha-beta T cell",  # ← new
    "mature NK T cell",
    "fibroblast",
    "endothelial cell",
    "smooth muscle cell",
    "acinar cell of salivary gland",  # ← new
    "duct epithelial cell",  # ← new
    "myoepithelial cell",  # ← new
    "ionocyte",  # ← new
    "pro-T cell",  # <- only 111 cells
]

# create filter_pass flag for Geneformer tokenization
adata.obs["filter_pass"] = [
    1 if ct in included_cell_types else 0 for ct in adata.obs["cell_type_union"]
]
adata.obs["filter_pass"].value_counts()

# %%
# DOWNSAMPLE ACINAR CELLS TO AVOID OVERREPRESENTATION
import numpy as np

np.random.seed(42)

acinar_mask = adata.obs["cell_type_union"] == "acinar cell of salivary gland"

# Split acinar cells by disease
acinar_normal_idx = adata.obs[acinar_mask & (adata.obs["disease"] == "normal")].index
acinar_sjogren_idx = adata.obs[
    acinar_mask & (adata.obs["disease"] == "Sjogren syndrome")
].index

print(f"Acinar normal: {len(acinar_normal_idx)}")
print(f"Acinar Sjogren: {len(acinar_sjogren_idx)}")

# Cap each at 4000 to keep acinar contribution balanced
keep_acinar_normal = np.random.choice(
    acinar_normal_idx, size=min(4000, len(acinar_normal_idx)), replace=False
)
keep_acinar_sjogren = np.random.choice(
    acinar_sjogren_idx, size=min(4000, len(acinar_sjogren_idx)), replace=False
)

non_acinar_pass = adata.obs[(adata.obs["filter_pass"] == 1) & ~acinar_mask].index

adata.obs["filter_pass"] = 0
adata.obs.loc[non_acinar_pass, "filter_pass"] = 1
adata.obs.loc[keep_acinar_normal, "filter_pass"] = 1
adata.obs.loc[keep_acinar_sjogren, "filter_pass"] = 1

print(adata.obs["filter_pass"].value_counts())
print(adata.obs[adata.obs["filter_pass"] == 1]["disease"].value_counts())

# After the acinar downsampling — also balance non-acinar cells by disease
non_acinar_passing = adata.obs[
    (adata.obs["filter_pass"] == 1)
    & (adata.obs["cell_type_union"] != "acinar cell of salivary gland")
]

non_acinar_normal = non_acinar_passing[non_acinar_passing["disease"] == "normal"].index
non_acinar_sjogren = non_acinar_passing[
    non_acinar_passing["disease"] == "Sjogren syndrome"
].index

print(f"Non-acinar normal: {len(non_acinar_normal)}")
print(f"Non-acinar Sjogren: {len(non_acinar_sjogren)}")

# Cap both at the size of the smaller group
target_n = min(len(non_acinar_normal), len(non_acinar_sjogren))
print(f"Capping each non-acinar disease group at: {target_n}")

keep_non_acinar_normal = np.random.choice(
    non_acinar_normal, size=target_n, replace=False
)
keep_non_acinar_sjogren = np.random.choice(
    non_acinar_sjogren, size=target_n, replace=False
)

# Rebuild filter_pass with balanced acinar + balanced non-acinar
adata.obs["filter_pass"] = 0
adata.obs.loc[keep_acinar_normal, "filter_pass"] = 1
adata.obs.loc[keep_acinar_sjogren, "filter_pass"] = 1
adata.obs.loc[keep_non_acinar_normal, "filter_pass"] = 1
adata.obs.loc[keep_non_acinar_sjogren, "filter_pass"] = 1

# Final sanity check
final = adata.obs[adata.obs["filter_pass"] == 1]
print("\n=== Final dataset ===")
print(final["disease"].value_counts())
print(f"Total cells: {len(final)}")

# %%
adata.write_h5ad(INPUT_DIR / "salivary-geneformer.h5ad")

# %% CUSTOM ATTRIBUTES
custom_attrs = {
    "cell_type_union": "cell-type",  # Here, this attribute is renamed
    "disease": "disease",
    "donor_id": "individual",  # Patient ID, later used for data splits
    "development_stage": "age",
    "sex": "sex",
}

# %% TOKENIZATION
from geneformer import TranscriptomeTokenizer

tk = TranscriptomeTokenizer(
    custom_attr_name_dict=custom_attrs,
    chunk_size=512,  # adjust based on available memory
    nproc=4,  # adjust based on available CPU cores
    model_version="V1",
)
tk.tokenize_data(
    data_directory=str(INPUT_DIR),
    output_directory=str(TOKENIZED_DIR),
    output_prefix="ss_tokenized",
    file_format="h5ad",
)
# %% FINE-TUNING
