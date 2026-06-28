# Annotate PBMC data for Sjogren's
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
import celltypist
import matplotlib.pyplot as plt
import mygene
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from celltypist.models import Model

# %%
adata = sc.read_h5ad(f"{DATA_DIR}/pbmc-processed.h5ad")

# %% RUN CELLTYPIST

# Diagnostic: Check if .X is raw counts or already processed
print("Min/max in .X before normalization:", adata.X.min(), adata.X.max())
print("Is .X integer type?", adata.X.dtype)

# If .X is not integer and min/max are not raw counts, reload from 10x
if not np.issubdtype(adata.X.dtype, np.integer) or adata.X.max() > 100:
    print("Reloading from 10x raw data...")
    adata = sc.read_10x_mtx(f"{RAW_DIR}", var_names="gene_symbols", cache=True)
    print("Min/max in .X after loading raw:", adata.X.min(), adata.X.max())

# Apply normalization and log1p-transform for CellTypist
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.X = np.nan_to_num(adata.X)
print("Min/max in .X after normalization:", adata.X.min(), adata.X.max())

# Map Ensembl IDs to gene symbols using mygene
mg = mygene.MyGeneInfo()
if "ensembl_id" in adata.var.columns:
    ensembl_ids = adata.var["ensembl_id"].tolist()
else:
    ensembl_ids = adata.var.index.tolist()
result = mg.querymany(
    ensembl_ids, scopes="ensembl.gene", fields="symbol", species="human"
)
symbol_map = {
    r["query"]: r.get("symbol") for r in result if "symbol" in r and r.get("symbol")
}
valid_genes = [eid for eid in ensembl_ids if eid in symbol_map]

print(f"Number of valid genes mapped: {len(valid_genes)} / {len(ensembl_ids)}")

if len(valid_genes) < 1000:
    print("Too few valid genes mapped. Using original gene symbols without filtering.")
    adata.var_names = (
        adata.var["gene_symbols"]
        if "gene_symbols" in adata.var.columns
        else adata.var.index
    )
else:
    adata = adata[:, valid_genes]
    adata.var_names = [symbol_map[eid] for eid in valid_genes]
    _, unique_idx = np.unique(adata.var_names, return_index=True)
    adata = adata[:, unique_idx]

adata.X = np.nan_to_num(adata.X)

# Run CellTypist annotation
model_dir = os.path.expanduser("~/.celltypist/data/models")
model_path = os.path.join(model_dir, "Immune_All_Low.pkl")
if os.path.exists(model_path):
    model = Model.load(model_path)
else:
    model = celltypist.models.download_models("Immune_All_Low.pkl")

predictions = celltypist.annotate(adata, model=model, majority_voting=True)
if "majority_voting" in predictions.predicted_labels.columns:
    adata.obs["celltypist_label"] = predictions.predicted_labels["majority_voting"]
else:
    adata.obs["celltypist_label"] = predictions.predicted_labels.iloc[:, 0]

adata.obs["celltypist_label"].to_csv(f"{OUTPUT_DIR}/celltypist_labels.csv")
adata.write_h5ad(f"{DATA_DIR}/pbmc-annotated.h5ad")

# %% CHECK ANNOTATIONS
# View unique cell types and counts
print("CellTypist label value counts:")
print(adata.obs["celltypist_label"].value_counts())

# Plot cell type distribution
plt.figure(figsize=(10, 5))
sns.countplot(
    y=adata.obs["celltypist_label"],
    order=adata.obs["celltypist_label"].value_counts().index,
)
plt.title("CellTypist Cell Type Distribution")
plt.xlabel("Number of Cells")
plt.ylabel("Cell Type")
plt.tight_layout()
plt.show()

# Cross-tabulate cell type by disease
if "disease" in adata.obs.columns:
    ctab = pd.crosstab(adata.obs["celltypist_label"], adata.obs["disease"])
    print("\nCell type by disease:")
    print(ctab)
    ctab.plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.title("Cell Type Distribution by Disease Status")
    plt.xlabel("Cell Type")
    plt.ylabel("Number of Cells")
    plt.tight_layout()
    plt.show()

# %% MEGE ANNOTATIONS WITH PROCESSED ANNDATA
celltypist_adata = sc.read_h5ad(f"{DATA_DIR}/pbmc-annotated.h5ad")
processed_adata = sc.read_h5ad(f"{DATA_DIR}/pbmc-processed.h5ad")

# Set index to barcodes for both AnnData objects if not already
if "barcodes" in celltypist_adata.obs.columns:
    celltypist_adata.obs.index = celltypist_adata.obs["barcodes"]
    celltypist_adata.obs.index.name = None
if "barcodes" in processed_adata.obs.columns:
    processed_adata.obs.index = processed_adata.obs["barcodes"]
    processed_adata.obs.index.name = None

print("First 5 rows of celltypist_adata.obs:")
print(celltypist_adata.obs.head())
print("First 5 rows of processed_adata.obs:")
print(processed_adata.obs.head())

# Diagnostics: Check barcode overlap between AnnData objects
barcodes_celltypist = set(celltypist_adata.obs.index)
barcodes_processed = set(processed_adata.obs.index)
print("Barcodes in celltypist AnnData:", len(barcodes_celltypist))
print("Barcodes in processed AnnData:", len(barcodes_processed))
print("Number of overlapping barcodes:", len(barcodes_celltypist & barcodes_processed))

# Check if celltypist_label exists and is populated in celltypist_adata
if (
    "celltypist_label" not in celltypist_adata.obs.columns
    or celltypist_adata.obs["celltypist_label"].isna().all()
):
    print(
        "celltypist_label missing or all NaN in celltypist_adata. Please rerun CellTypist annotation."
    )
else:
    print("celltypist_label value counts in celltypist_adata:")
    print(celltypist_adata.obs["celltypist_label"].value_counts(dropna=False).head(20))
    # Merge CellTypist labels
    celltypist_labels = celltypist_adata.obs["celltypist_label"]
    if "celltypist_label" in processed_adata.obs.columns:
        processed_adata.obs = processed_adata.obs.drop(columns=["celltypist_label"])
    processed_adata.obs = processed_adata.obs.join(celltypist_labels, how="left")
    print("Processed AnnData celltypist_label value counts after merge:")
    print(processed_adata.obs["celltypist_label"].value_counts(dropna=False))
    processed_adata.write_h5ad(f"{DATA_DIR}/pbmc-annotated.h5ad")
