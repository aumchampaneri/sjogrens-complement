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
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from adpbulk import ADPBulk
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

# %%
adata = sc.read_h5ad(f"{DATA_DIR}/pbmc-annotated.h5ad")

# %% PSEUDOBULK DATA
adpb = ADPBulk(adata, ["celltypist_label", "cell_batch"], use_raw=True)
pseudobulk_matrix = adpb.fit_transform()
sample_meta = adpb.get_meta()

# save pseudobulk matrix and metadata
pseudobulk_matrix.to_csv(f"{OUTPUT_DIR}/pbmc-pseudobulk_matrix.csv")
sample_meta.to_csv(f"{OUTPUT_DIR}/pbmc-pseudobulk_metadata.csv")

print("Pseudobulk matrix shape:", pseudobulk_matrix.shape)
display(pseudobulk_matrix.head())
print("Sample metadata:")
display(sample_meta.head())

# %% RUN PYDESEQ2

# Load pseudobulk matrix and metadata
counts = pd.read_csv(f"{OUTPUT_DIR}/pbmc-pseudobulk_matrix.csv", index_col=0)
meta = pd.read_csv(f"{OUTPUT_DIR}/pbmc-pseudobulk_metadata.csv", index_col=0)

# Set metadata index to SampleName column for alignment
if "SampleName" in meta.columns:
    meta = meta.set_index("SampleName")

# Extract disease from cell_batch if not present
if "disease" not in meta.columns and "cell_batch" in meta.columns:
    meta["disease"] = (
        meta["cell_batch"].str[:2].map({"HC": "normal", "pS": "sjogren syndrome"})
    )

# Align meta to counts index (samples)
meta_aligned = meta.reindex(counts.index)

# Drop samples with missing metadata
missing = meta_aligned.isnull().any(axis=1)
if missing.any():
    print(
        "Warning: Missing metadata for samples:", meta_aligned.index[missing].tolist()
    )
    counts = counts.loc[~missing, :]
    meta_aligned = meta_aligned.loc[~missing]

# Convert counts to integer if needed
if not all(counts.dtypes == "int"):
    counts = counts.round().astype(int)

# Optional: use multi-core inference
inference = DefaultInference(n_cpus=8)

# Stepwise fitting for transparency and reproducibility
dds = DeseqDataSet(
    counts=counts,
    metadata=meta_aligned,
    design_factors=["disease"],
    refit_cooks=True,
    inference=inference,
    # design="~disease"  # Uncomment if you want to use formula syntax
    # control_genes=None # Optionally specify control genes for normalization
)
dds.fit_size_factors()
print("Size factors:")
print(dds.obs["size_factors"].head())
dds.fit_genewise_dispersions()
print("Genewise dispersions:")
print(dds.var["genewise_dispersions"].head())
dds.fit_dispersion_trend()
print("Dispersion trend coefficients:")
# print(dds.uns["trend_coeffs"])
dds.fit_dispersion_prior()
print(
    f"logres_prior={dds.uns['_squared_logres']}, sigma_prior={dds.uns['prior_disp_var']}"
)
dds.fit_MAP_dispersions()
print("MAP dispersions:")
print(dds.var["MAP_dispersions"].head())
dds.fit_LFC()
print("LFCs (natural log scale):")
print(dds.varm["LFC"][:5])
dds.calculate_cooks()
if dds.refit_cooks:
    dds.refit()
print("Cooks distances calculated and outliers refit if needed.")

# Save dds object for reproducibility
with open("dds_detailed_pipe.pkl", "wb") as f:
    pkl.dump(dds, f)

# Statistical analysis with DeseqStats
ds = DeseqStats(
    dds,
    contrast=["disease", "sjogren syndrome", "normal"],
    alpha=0.05,
    cooks_filter=True,
    independent_filter=True,
)
ds.run_wald_test()
if ds.cooks_filter:
    ds._cooks_filtering()
if ds.independent_filter:
    ds._independent_filtering()
else:
    ds._p_value_adjustment()
print("Adjusted p-values:")
print(ds.padj.head())
ds.summary()
res_df = ds.results_df
print(res_df.head())
# LFC shrinkage (recommended for visualization)
ds.lfc_shrink(coeff="disease[T.sjogren syndrome]")
print("Shrunk log2 fold changes:")
print(ds.results_df.head())
# Save ds object for reproducibility
with open("ds_detailed_pipe.pkl", "wb") as f:
    pkl.dump(ds, f)

# Save DESeq2 results
res_df.to_csv(f"{OUTPUT_DIR}/pbmc-deseq2_results.csv")
# %% PLOT PYDESEQ2 RESULTS

# Load complement gene list
with open(f"{RESOURCE_DIR}/complement-genes.txt") as f:
    complement_genes = [line.strip() for line in f if line.strip()]

# Filter DESeq2 results for complement genes
complement_res = res_df[res_df.index.isin(complement_genes)].copy()

# Sort by adjusted p-value
complement_res = complement_res.sort_values("padj")

# %% Plot log2 fold change for complement genes
plt.figure(figsize=(10, 10))
sns.barplot(y=complement_res.index, x=complement_res["log2FoldChange"], palette="vlag")
plt.xlabel("log2 Fold Change (sjogren syndrome vs normal)")
plt.ylabel("Gene")
plt.title("DESeq2 log2 Fold Change for Complement Genes")
plt.tight_layout()
plt.show()

# %% Volcano plot for complement genes
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=complement_res["log2FoldChange"],
    y=-np.log10(complement_res["padj"]),
    hue=(complement_res["padj"] < 0.05),
    palette={True: "red", False: "grey"},
    legend=False,
)
for gene, row in complement_res.iterrows():
    if row["padj"] < 0.05:
        plt.text(row["log2FoldChange"], -np.log10(row["padj"]), gene, fontsize=8)
plt.xlabel("log2 Fold Change")
plt.ylabel("-log10(padj)")
plt.title("Volcano Plot: Complement Genes")
plt.tight_layout()
plt.show()

# %% MA Plot for all genes
# Ensure DESeq2 results are loaded into res_df before plotting
# If you have a DeseqStats object named 'ds', use the following line:
res_df = ds.results_df

plt.figure(figsize=(8, 6))
plt.scatter(
    res_df["baseMean"],
    res_df["log2FoldChange"],
    c=(res_df["padj"] < 0.05),
    cmap="coolwarm",
    alpha=0.5,
    edgecolor="none",
)
plt.xscale("log")
plt.xlabel("Mean Expression (baseMean, log scale)")
plt.ylabel("log2 Fold Change")
plt.title("MA Plot: All Genes")
plt.tight_layout()
plt.show()

# %% Heatmap of top differentially expressed complement genes
# Select top 20 by adjusted p-value
n_top = 20
top_complement = complement_res.nsmallest(n_top, "padj")
# Load normalized counts if available, else use counts
try:
    norm_counts = dds.norm_counts
except AttributeError:
    norm_counts = counts
# If genes are columns, transpose so genes are rows
genes_in_columns = set(top_complement.index).issubset(norm_counts.columns)
if genes_in_columns:
    heatmap_data = norm_counts[top_complement.index].T
else:
    heatmap_data = norm_counts.loc[top_complement.index]
# Z-score normalization for visualization (pandas broadcasting, no keepdims)
heatmap_data = (heatmap_data.subtract(heatmap_data.mean(axis=1), axis=0)).divide(
    heatmap_data.std(axis=1), axis=0
)
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap="vlag", yticklabels=top_complement.index)
plt.title("Heatmap: Top Differentially Expressed Complement Genes")
plt.xlabel("Sample")
plt.ylabel("Gene")
plt.tight_layout()
plt.show()
# %% Lollipop plot: complement genes ranked by -log10(padj)
complement_res["-log10_padj"] = -np.log10(complement_res["padj"])
complement_res_sorted = complement_res.sort_values("-log10_padj", ascending=False)
plt.figure(figsize=(10, 8))
# Removed use_line_collection for compatibility with newer matplotlib
plt.stem(complement_res_sorted.index, complement_res_sorted["-log10_padj"], basefmt=" ")
plt.xticks(rotation=90)
plt.ylabel("-log10(padj)")
plt.xlabel("Gene")
plt.title("Lollipop Plot: Complement Genes by Significance")
plt.tight_layout()
plt.show()

# %% Boxplots for selected complement genes (top 5 by significance)
# Check if genes are in index or columns
top5 = complement_res.nsmallest(5, "padj").index.tolist()
if set(top5).issubset(norm_counts.index):
    plot_data = (
        norm_counts.loc[top5]
        .T.reset_index()
        .melt(id_vars="index", var_name="Gene", value_name="Expression")
    )
elif set(top5).issubset(norm_counts.columns):
    plot_data = (
        norm_counts[top5]
        .reset_index()
        .melt(id_vars="index", var_name="Gene", value_name="Expression")
    )
else:
    raise KeyError(f"None of {top5} are in the index or columns of norm_counts")
# Add disease group info
plot_data = plot_data.rename(columns={"index": "Sample"})
if "disease" in meta_aligned.columns:
    plot_data = plot_data.merge(
        meta_aligned[["disease"]], left_on="Sample", right_index=True, how="left"
    )
# Log-transform expression (add small constant to avoid log(0))
plot_data["log_Expression"] = np.log1p(plot_data["Expression"])
plt.figure(figsize=(12, 6))
sns.boxplot(x="Gene", y="log_Expression", hue="disease", data=plot_data)
plt.title("Boxplots: Top 5 Complement Genes by Disease Group (log scale)")
plt.ylabel("log(Expression + 1)")
plt.tight_layout()
plt.show()
