# Prepare, process, and scVI
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

RESOURCE_DIR = PROJECT_DIR / "resources/"

# %% IMPORTS
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
import torch

# %%
adata = sc.read_h5ad(f"{DATA_DIR}/salivary-cxg.h5ad")
# %%
# PREPARE DATA
# Modify and reformat any oddly formatted columns

# Duplicate index to a new .var column -> "ensembl_id"
adata.var["ensembl_id"] = adata.var.index

# Sjogren syndrome & normal (control)
set(adata.obs["disease"])

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

# sanity check
set(adata.obs["cell_type_union"])

# reformat developmental stage / age attribute
set(adata.obs["development_stage"])

# keep only first two characters (e.g., "Adult" -> "Ad")
adata.obs["development_stage"] = [ds[:2] for ds in adata.obs["development_stage"]]

# sanity check
set(adata.obs["development_stage"])
# %%

adata.write_h5ad(f"{DATA_DIR}/salivary-prepared.h5ad")

# %%
# PROCESS DATA
# Loads prepared data for raw count processing using scVI
# Performs QC and defines 'custom_attrs'
adata = sc.read_h5ad(f"{DATA_DIR}/salivary-prepared.h5ad")

# %%
# QUALITY CONTROL

# mitochondrial genes
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# filter extreme cells
adata = adata[adata.obs.pct_counts_mt < 20, :]
adata = adata[adata.obs.n_genes_by_counts > 200, :]

print(adata.raw)
print("X shape:", adata.X.shape)
print("raw.X shape:", adata.raw.X.shape)


def summarize(matrix, name):
    print(f"\n{name}")
    print("min:", matrix.min())
    print("max:", matrix.max())
    print("mean:", matrix.mean())
    print("non-zero fraction:", (matrix > 0).mean())


# main matrix
summarize(adata.X.A if hasattr(adata.X, "A") else adata.X, "adata.X")

# raw matrix
summarize(adata.raw.X.A if hasattr(adata.raw.X, "A") else adata.raw.X, "adata.raw.X")

print(
    "Any negative values in raw?",
    (adata.raw.X < 0).sum() if hasattr(adata.raw.X, "A") else np.sum(adata.raw.X < 0),
)

# %%
# SET CUSTOM ATTRIBUTES FOR DOWNSTREAM
custom_attrs = {
    "cell_type_union": "cell-type",  # Here, this attribute is renamed
    "disease": "disease",
    "donor_id": "individual",  # Patient ID, later used for data splits
    "development_stage": "age",
    "sex": "sex",
}

# %%
# scVI
#  Define input and train scVI
# operate on a copy
adata = adata.copy()

# create 'counts' layer from raw counts if it doesn't exist
if "counts" not in adata.layers:
    if getattr(adata, "raw", None) is None:
        raise RuntimeError(
            "adata.raw is None — no raw counts available. Set adata.raw or provide counts in adata.layers."
        )
    counts = adata.raw.X
    if sp.issparse(counts):
        counts = counts.tocsr().astype(np.float32)
    else:
        counts = np.array(counts, dtype=np.float32, copy=True)
    adata.layers["counts"] = counts

print(
    "counts layer shape:",
    adata.layers["counts"].shape,
    "dtype:",
    getattr(adata.layers["counts"], "dtype", None),
)

# auto-detect a plausible batch key (optional)
batch_key = None
for c in ("donor_id", "sample_id", "donor", "batch", "individual"):
    if c in adata.obs.columns:
        batch_key = c
        print("Using batch_key:", batch_key)
        break

# setup anndata for scvi using counts
if batch_key:
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
else:
    scvi.model.SCVI.setup_anndata(adata, layer="counts")

# pick accelerator (mps/gpu/cpu)
accel = "mps"
devices = 4
try:
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        accel = "mps"
        devices = 1
    elif torch.cuda.is_available():
        accel = "gpu"
        devices = 1
except Exception:
    pass

model = scvi.model.SCVI(adata)
model.train(max_epochs=200, accelerator=accel, devices=devices)

adata.obsm["X_scVI"] = model.get_latent_representation()

# %%
# Neighbors and UMAP (scVI)
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

# %%
# Plots
# adata = sc.read_h5ad(f"{OUTPUT_DIR}/salivary-scVI.h5ad")
sc.pl.umap(adata, color=["leiden"], legend_loc="on data")
sc.pl.umap(adata, color=["cell_type"])
sc.pl.umap(adata, color=["donor_id"])
sc.pl.umap(adata, color=["disease"])

# %%
# Save Results
adata.write_h5ad(f"{DATA_DIR}/salivary-scvi.h5ad")
model.save(f"{DATA_DIR}/scvi_model/", overwrite=True)

# %%
# Figures
# adata = sc.read_h5ad(f"{OUTPUT_DIR}/salivary-scVI.h5ad")

# complement list (from resources if present)
comp_path = os.path.join(RESOURCE_DIR, "resources", "complement-genes.txt")
if os.path.exists(comp_path):
    with open(comp_path) as f:
        comp_genes = [l.strip() for l in f if l.strip() and not l.startswith("#")]
else:
    comp_genes = [
        "C1QA",
        "C1QB",
        "C1QC",
        "C2",
        "C3",
        "C4A",
        "C4B",
        "C5",
        "C6",
        "C7",
        "C8A",
        "C8B",
        "C8G",
        "C9",
    ]

print("Complement genes:", comp_genes)

filt_comp_genes = [
    "C1QA",
    "C1QB",
    "C1QC",
    "C2",
    "C3",
    "C3AR1",
    "C5",
    "C5AR1",
    "C5AR2",
    "C1R",
    "C1S",
    "CD46",
    "CD55",
    "CD59",
    "CFB",
    "CFD",
    "CFH",
    "CFI",
    "CFP",
    "CLU",
    "ITGAM",
    "ITGAX",
    "PROS1",
    "SERPING1",
    "THBD",
    "VSIG4",
]

sc.pl.dotplot(
    adata,
    var_names=comp_genes,
    gene_symbols="feature_name",
    groupby="cell_type_union",
    use_raw=False,
    log=False,
    mean_only_expressed=True,
    standard_scale="var",
    show=True,
)

sc.pl.dotplot(
    adata,
    var_names=filt_comp_genes,
    gene_symbols="feature_name",
    groupby="cell_type_union",
    use_raw=False,
    log=False,
    mean_only_expressed=True,
    standard_scale="var",
    show=True,
)

sc.pl.dotplot(
    adata,
    var_names=filt_comp_genes,
    gene_symbols="feature_name",
    groupby="cell_type_union",
    use_raw=False,
    log=False,
    mean_only_expressed=True,
    standard_scale="var",
    show=True,
    dendrogram=True,
)

sc.pl.umap(adata, color="cell_type_union", show=True, title=" ")

sc.pl.umap(
    adata,
    color=["C1QA", "C3", "CFH", "C5AR1", "C5", "C3AR1", "C1R", "CD59"],
    gene_symbols="feature_name",
    use_raw=False,
    show=True,
    color_map="magma",
)

sc.pl.umap(
    adata,
    color="C1QA",
    gene_symbols="feature_name",
    use_raw=False,
    show=True,
    color_map="magma",
)


def plot_ct_composition(
    adata, sample_col="Sample", cell_col="cell_type", disease_col="disease"
):
    if cell_col not in adata.obs.columns or sample_col not in adata.obs.columns:
        raise RuntimeError("Missing sample or cell-type column")

    ctab = pd.crosstab(
        adata.obs[sample_col], adata.obs[cell_col], normalize="index"
    ).fillna(0)

    if disease_col in adata.obs.columns:
        samp2dis = adata.obs.groupby(sample_col)[disease_col].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
        )
        common = ctab.index.intersection(samp2dis.index)
        ctab = ctab.loc[common]
        samp2dis = samp2dis.loc[common]
        ctab = ctab.loc[samp2dis.sort_values().index]

    ax = ctab.plot(
        kind="bar",
        stacked=True,
        colormap="tab20",
        figsize=(14, 5),
        width=0.8,
        legend=False,
    )
    ax.set(xlabel="Sample", ylabel="Fraction", title="Cell-type composition per sample")
    plt.legend(bbox_to_anchor=(1, 1), title="cell type")
    plt.tight_layout()

    if disease_col in adata.obs.columns:
        agg = ctab.groupby(samp2dis).mean()
        agg.plot(kind="bar", stacked=True, colormap="tab20", figsize=(6, 4), width=0.8)
        plt.ylabel("Fraction")
        plt.title("Mean cell-type composition by disease")
        plt.tight_layout()


plot_ct_composition(
    adata, sample_col="donor_id", cell_col="cell_type_union", disease_col="disease"
)
