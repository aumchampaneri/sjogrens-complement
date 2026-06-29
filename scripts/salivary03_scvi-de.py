# scVI Differential Expression
#
#
# %%
from pathlib import Path

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "outputs" / "salivary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = PROJECT_DIR / "outputs" / "salivary" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_DIR / "data" / "salivary"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RESOURCE_DIR = PROJECT_DIR / "resources"

# %% IMPORTS
import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
import seaborn as sns
import sympy
import torch

# %%
adata = sc.read_h5ad(DATA_DIR / "salivary-scvi.h5ad")
model = scvi.model.SCVI.load(
    DATA_DIR / "scvi_model",
    adata=adata,
    accelerator="mps" if use_mps else "cpu",
    device=1 if use_mps else None,
)
device = "mps"
model.to_device(device)

# %% Load Complement Genes
comp_path = RESOURCE_DIR / "complement-genes.txt"
with open(comp_path, "r") as f:
    comp_genes = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]
print("Complement genes:", comp_genes)
# %%
# check docstring first: help(model.differential_expression)
de_df = model.differential_expression(
    groupby="disease",
    group1="Sjogren syndrome",
    group2="normal",
    # weights="importance",
    filter_outlier_cells=True,
    batch_correction=True,  # include if you want to correct batches
)
src_csv = OUTPUT_DIR / "scvi_de_sjogren_vs_normal.csv"
de_df.to_csv(src_csv, index=True)

# %%
# Map ENSEMBL -> gene symbols for full DE table
mapped_csv = OUTPUT_DIR / "scvi_de_sjogren_vs_normal.mapped.csv"

if "de_df" not in globals() or "adata" not in globals():
    raise RuntimeError(
        "Required data states ('de_df' or 'adata') are missing. Run previous cells first."
    )

df = de_df.copy()
var = adata.var

# Clean ENSEMBL IDs to remove version suffixes (e.g., ENSG000001.2 -> ENSG000001)
if "ensembl_id" in var.columns:
    ensembl_series = var["ensembl_id"].astype(str).str.split(".").str[0]
else:
    ensembl_series = var.index.astype(str).str.split(".").str[0]

mapping = dict(zip(ensembl_series, var["feature_name"]))

clean_df_index = pd.Series(df.index.astype(str).str.split(".").str[0], index=df.index)
df["gene_symbol"] = clean_df_index.map(mapping).fillna(df.index.to_series())

df.to_csv(mapped_csv)
print(f"Saved mapped DE table to: {mapped_csv}")
