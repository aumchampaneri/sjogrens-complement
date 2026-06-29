# Pseudobulk and perform DE (pyDeSeq2) PBMC data for Sjogren's
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

RESOURCE_DIR = PROJECT_DIR / "resources/"

RAW_DIR = PROJECT_DIR / "data/pbmc/raw/"

# %% IMPORTS
import warnings

warnings.filterwarnings("ignore")

from collections import defaultdict

import cell2cell as c2c
import decoupler as dc
import liana as li
import mygene
import numpy as np
import scanpy as sc
import tensorly as tl
import torch

# Set-up for Tensor-cell2cell
if torch.backends.mps.is_available():
    device = "mps"
    tl.set_backend("pytorch")
elif torch.cuda.is_available():
    device = "cuda"
    tl.set_backend("pytorch")
else:
    device = "cpu"
    tl.set_backend("pytorch")

print("Using device:", device)

# %%
adata = sc.read_h5ad(f"{DATA_DIR}/pbmc-annotated.h5ad")

# %%
# Map Ensembl IDs to gene symbols for LIANA compatibility
# Only run if not already mapped
if "gene_symbols" not in adata.var.columns or adata.var["gene_symbols"].isnull().all():
    mg = mygene.MyGeneInfo()
    ensembl_ids = adata.var.index.tolist()
    print(f"Querying mygene for {len(ensembl_ids)} Ensembl IDs...")
    result = mg.querymany(
        ensembl_ids, scopes="ensembl.gene", fields="symbol", species="human"
    )
    symbol_map = {
        r["query"]: r.get("symbol") for r in result if "symbol" in r and r.get("symbol")
    }
    adata.var["gene_symbols"] = adata.var.index.map(symbol_map)

# Filter out genes without a symbol
valid = adata.var["gene_symbols"].notnull()
adata = adata[:, valid].copy()

# Set gene symbols as var_names for LIANA.
# AnnData slicing is strict about index dtypes, so force plain string names here.
adata.var_names = adata.var["gene_symbols"].astype(str)
adata.var_names_make_unique()
print("First 10 gene symbols:", adata.var_names[:10])

# Ensure required columns exist and are categorical
for col in ["celltypist_label", "batch", "disease"]:
    if col not in adata.obs.columns:
        raise ValueError(
            f"Column '{col}' missing in adata.obs. Please check your annotation steps."
        )
    adata.obs[col] = adata.obs[col].astype("category")

# For LIANA, set cell type, sample, and condition columns
adata.obs["cell_type"] = adata.obs["celltypist_label"]
adata.obs["sample"] = adata.obs["batch"]
adata.obs["condition"] = adata.obs["disease"]

print("adata.obs columns:", adata.obs.columns.tolist())
print("Unique cell types:", adata.obs["cell_type"].unique())
print("Unique samples:", adata.obs["sample"].unique())
print("Unique conditions:", adata.obs["condition"].unique())

# %%
# Run LIANA by sample (consensus resource, robust rank aggregation)
li.mt.rank_aggregate.by_sample(
    adata,
    groupby="cell_type",
    resource_name="consensus",
    sample_key="sample",
    use_raw=True,
    verbose=True,
    n_perms=None,
    return_all_lrs=True,
)

# Save LIANA results for reproducibility
liana_outfile = os.path.join(OUTPUT_DIR, "liana_by_sample.csv")
adata.uns["liana_res"].to_csv(liana_outfile)
print(f"Saved LIANA results to {liana_outfile}")

# %% Build the Tensor
# Build tensor for Tensor-cell2cell using a valid score column
tensor = li.multi.to_tensor_c2c(
    adata,
    sample_key="sample",
    score_key="magnitude_rank",  # or 'lrscore' if you prefer
    how="outer_cells",
)
print("Tensor shape (Contexts, Interactions, Senders, Receivers):", tensor.tensor.shape)

# Save tensor object
tensor_outfile = os.path.join(PDIR, "sjogrens-pbmc/outputs/liana_tensor.pkl")
c2c.io.export_variable_with_pickle(tensor, tensor_outfile)
print(f"Saved tensor to {tensor_outfile}")

# check the shape of the tensor, represented as (Contexts, Interactions, Senders, Receivers).
tensor.tensor.shape

# Build metadata for tensor decomposition
context_dict = adata.obs[["sample", "condition"]].drop_duplicates()
context_dict = dict(zip(context_dict["sample"], context_dict["condition"]))
context_dict = defaultdict(lambda: "Unknown", context_dict)

tensor_meta = c2c.tensor.generate_tensor_metadata(
    interaction_tensor=tensor,
    metadata_dicts=[context_dict, None, None, None],
    fill_with_order_elements=True,
)

# %% Run Tensor-cell2cell
# Run Tensor-cell2cell decomposition (auto rank estimation, can set rank=int for speed/reproducibility)
tensor = c2c.analysis.run_tensor_cell2cell_pipeline(
    tensor,
    tensor_meta,
    copy_tensor=None,
    rank=9,  # "The rank at the elbow is 9"
    tf_optimization="regular",
    random_state=42,
    device=device,
    elbow_metric="error",
    smooth_elbow=False,
    upper_rank=20,
    tf_init="random",
    tf_svd="numpy_svd",
    cmaps=None,
    sample_col="Element",
    group_col="Category",
    output_fig=False,
)

# Plot tensor factors (latent CCC patterns)
factors, axes = c2c.plotting.tensor_factors_plot(
    interaction_tensor=tensor,
    metadata=tensor_meta,
    sample_col="Element",
    group_col="Category",
    meta_cmaps=["viridis", "Dark2_r", "tab20", "tab20"],
    fontsize=10,
)

# %% Factorization Results
# Access factors and loadings
factors = tensor.factors
print(
    factors.keys()
)  # Should show Contexts, Ligand-Receptor Pairs, Sender Cells, Receiver Cells

# Show top ligand-receptor pairs for a factor (e.g., Factor 1)
lr_loadings = factors["Ligand-Receptor Pairs"]
lr_loadings.sort_values("Factor 1", ascending=False).head(10)

# %%
# Visualize context loadings (boxplot)
_ = c2c.plotting.context_boxplot(
    context_loadings=factors["Contexts"],
    metadict=context_dict,  # Make sure context_dict maps context names to metadata
    nrows=2,
    figsize=(14, 8),
    statistical_test="t-test_ind",
    pval_correction="fdr_bh",
    cmap="plasma",
    verbose=False,
)
# %%
# Visualize CCC networks for a factor of interest (e.g., Factor 1 or 2, adjust as needed)
_ = c2c.plotting.ccc_networks_plot(
    factors,
    included_factors=["Factor 9"],  # Change to your factor of interest
    network_layout="circular",
    ccc_threshold=0.05,
    nrows=1,
    panel_size=(12, 12),
)
# %%
# Optional: Pathway enrichment analysis for factors
# Load PROGENy pathways
net = dc.op.progeny(organism="human", top=5000)
# Load full list of ligand-receptor pairs
lr_pairs = li.resource.select_resource("consensus")
# Generate ligand-receptor geneset
lr_progeny = li.rs.generate_lr_geneset(lr_pairs, net, lr_sep="^").rename(
    columns={"interaction": "target"}
)
lr_progeny.head()
# Run enrichment
estimate, pvals = dc.mt.ulm(
    tensor.factors["Ligand-Receptor Pairs"].transpose(), lr_progeny, raw=False
)
# Plot enrichment for a factor (e.g., 'Factor 1')
dc.pl.barplot(estimate, "Factor 3", vertical=True, cmap="coolwarm", vmin=-7, vmax=7)
