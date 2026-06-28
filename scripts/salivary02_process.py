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
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import scvi
import torch

# %%
adata = sc.read_h5ad(f"{DATA_DIR}/cxg_sjogrens.h5ad")
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

adata.write_h5ad(f"{DATA_DIR}/cxg_sjogrens_prepared.h5ad")

# %%
# PROCESS DATA
# Loads prepared data for raw count processing using scVI
# Performs QC and defines 'custom_attrs'
adata = sc.read_h5ad(f"{DATA_DIR}/cxg_sjogrens_prepared.h5ad")

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
# adata = sc.read_h5ad(f"{OUTPUT_DIR}/cxg_sjogrens_scVI.h5ad")
sc.pl.umap(adata, color=["leiden"], legend_loc="on data")
sc.pl.umap(adata, color=["cell_type"])
sc.pl.umap(adata, color=["donor_id"])
sc.pl.umap(adata, color=["disease"])

# %%
# Figures
# adata = sc.read_h5ad(f"{OUTPUT_DIR}/cxg_sjogrens_scVI.h5ad")

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
