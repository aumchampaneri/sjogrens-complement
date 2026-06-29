# Process PBMC data for Sjogren's
#
#
# %%
import os
from pathlib import Path

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "outputs/pbmc/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_DIR / "data/pbmc/"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = PROJECT_DIR / "data/pbmc/raw/"

# %% IMPORTS
import gzip
import shutil

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce

# %% CONVERT RAW DATA TO ANNDATA
print("Files in directory:", os.listdir(RAW_DIR))

# Read 10x Genomics formatted data
adata = sc.read_10x_mtx(
    RAW_DIR,  # Path to directory with matrix.mtx, barcodes.tsv, features.tsv
    var_names="gene_symbols",  # Use gene symbols for variable names
    cache=True,  # Cache the result for faster future loading
)

# Set .raw attribute to preserve raw counts for downstream pseudobulk/DESeq2
adata.raw = adata.copy()
print("Set adata.raw. Shape:", adata.raw.shape)

# Unpack cell_batch.tsv.gz if it exists and append to adata.obs
cell_batch_path = os.path.join(RAW_DIR, "cell_batch.tsv.gz")
if os.path.exists(cell_batch_path):
    with gzip.open(cell_batch_path, "rt") as f_in:
        with open("cell_batch.tsv", "w") as f_out:
            shutil.copyfileobj(f_in, f_out)
    cell_batch_df = pd.read_csv("cell_batch.tsv", sep="\t", index_col=0)
    adata.obs = adata.obs.join(cell_batch_df)
    os.remove("cell_batch.tsv")  # Clean up the unzipped file
    print("Appended cell batch information to adata.obs")

# %% DATA EXPLORATION
adata
# %%
adata.obs
# %%
adata.var

# %% DATA MODIFICATION
# 1. Merge 'cell_batch.tsv' to .obs layer
# 2. Extrapolate 'disease' and patient_ID
# 3. Rearange and rename .var columns

# Read the cell_batch file, using the first column (barcodes) as the index
cell_batch = pd.read_csv(
    f"{RAW_DIR}/cell_batch.tsv.gz", sep="\t", header=0, index_col=0
)

# Align and assign the batch/condition info to AnnData obs
adata.obs["cell_batch"] = adata.obs_names.map(cell_batch.iloc[:, 0])

# Preview the result
print(adata.obs[["cell_batch"]].head())

# Duplicate barcodes index into a new column
adata.obs["barcodes"] = adata.obs.index
print("adata.obs columns after adding barcodes:", adata.obs.columns)
print(adata.obs.head())

# Extract disease and patient id from the cell_batch column
disease_labels = adata.obs["cell_batch"].str.extract(r"^(pSS|HC)")[0]
patient_ids = adata.obs["cell_batch"].str.extract(r"-(\d+)$")[0]

# Assign disease column
adata.obs["disease"] = disease_labels

# Set HC in disease column to 'normal'
adata.obs.loc[adata.obs["disease"] == "HC", "disease"] = "normal"

# Set pSS in disease column to 'sjogren syndrome'
adata.obs.loc[adata.obs["disease"] == "pSS", "disease"] = "sjogren syndrome"

# Create a mapping from (disease, patient_id) to a unique join_id number (1-10)
unique_patients = adata.obs[["disease", "cell_batch"]].drop_duplicates()
unique_patients["join_id"] = range(1, len(unique_patients) + 1)

# Merge back to obs to assign patient numbers
adata.obs = adata.obs.merge(
    unique_patients[["cell_batch", "join_id"]],
    left_on="cell_batch",
    right_on="cell_batch",
    how="left",
)

# Add count data
adata.obs["n_counts"] = adata.X.sum(axis=1)

# Assign batch to cell_batch for pseudobulk aggregation
adata.obs["batch"] = adata.obs["cell_batch"]

# Preview the new columns
print(adata.obs[["cell_batch", "disease", "join_id", "batch"]].head())

# Duplicate the .var index to a new column 'feature_names'
adata.var["feature_names"] = adata.var.index

# Rename gene_ids to ensembl_id
adata.var.rename(columns={"gene_ids": "ensembl_id"}, inplace=True)

# Replace the information in the index column with the ensembl_id values
adata.var.index = adata.var["ensembl_id"]

# Clear the index column name
adata.var.index.name = None

adata.write_h5ad(f"{DATA_DIR}/pbmc-prepared.h5ad")

# %% DATA PROCESSING AND QC
# 1. Load adata
# 2. scanpy zheng17 recipe
# 3. UMAP
# 4. Save adata

adata = sc.read_h5ad(f"{DATA_DIR}/pbmc-prepared.h5ad")

sc.pp.recipe_zheng17(adata, n_top_genes=20000, log=True, plot=True, copy=False)

sc.pp.pca(adata, n_comps=50, svd_solver="arpack")
sce.pp.harmony_integrate(adata, "batch")
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)

sc.tl.umap(adata)
sc.tl.leiden(adata)

sc.pl.pca(adata, color="batch")
sc.pl.umap(adata, color=["batch", "disease", "leiden"])

adata.write_h5ad(f"{DATA_DIR}/pbmc-processed.h5ad")
