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

FINETUNE_DIR = DATA_DIR / "geneformer" / "finetune"
FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

EMB_DIR = DATA_DIR / "geneformer" / "emb"
EMB_DIR.mkdir(parents=True, exist_ok=True)

RESOURCE_DIR = PROJECT_DIR / "resources"

GENEFORMER = PROJECT_DIR / "geneformer"
GENEFORMER.mkdir(parents=True, exist_ok=True)

# %% IMPORTS
import datetime
import subprocess

import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split

from geneformer import Classifier, EmbExtractor, TranscriptomeTokenizer

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
current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = (
    f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
)
output_prefix = "ss_dz_classifier"
output_dir = FINETUNE_DIR / f"{datestamp}"
os.makedirs(output_dir, exist_ok=True)
print(datestamp)

# %% TRAINING ARGUMENTS
training_args = {
    "num_train_epochs": 3,  # Faster iteration
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,  # Slightly higher for faster convergence in small sets
    "lr_scheduler_type": "linear",
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "label_smoothing_factor": 0.1,
    "eval_strategy": "epoch",
    "load_best_model_at_end": True,
}

# --- Stratified Subset for Speed ---
# Use this to quickly test if your model can learn BEFORE doing the 5hr run
subset_indices = []
for disease in adata.obs["disease"].unique():
    # Take 2000 cells per disease, stratified by donor where possible
    subset = adata.obs[adata.obs["disease"] == disease].sample(n=2000, random_state=42)
    subset_indices.extend(subset.index)

adata_subset = adata[subset_indices].copy()

cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": "all"},
    filter_data={
        "cell-type": [
            "Macrophage",
            "dendritic cell",
            "B cell",
            "Plasma cell",
            "CD4-positive, alpha-beta T cell",
            "CD8-positive, alpha-beta cytotoxic T cell",
            "CD8-positive, alpha-beta regulatory T cell",  # autoimmune relevance
            "effector CD8-positive, alpha-beta T cell",  # tissue inflammation
            "mature NK T cell",
            "fibroblast",
            "endothelial cell",
            "smooth muscle cell",
            "acinar cell of salivary gland",  # primary target tissue in Sjögren's
            "duct epithelial cell",  # also affected in Sjögren's
            "myoepithelial cell",  # salivary gland structural cell
            "ionocyte",
        ]
    },
    training_args=training_args,
    max_ncells=None,  # use all cells
    freeze_layers=2,
    num_crossval_splits=1,
    forward_batch_size=16,
    nproc=4,  # ← back to 4
    ngpu=0,
    model_version="V1",
)

# %% SPLIT DATASET INTO TRAINING-VALIDATION-TEST SETS
# 1. Isolate cells that passed your initial filters
passing_cells_idx = adata.obs[adata.obs["filter_pass"] == 1].index
adata_filtered = adata[passing_cells_idx].copy()

# 2. Get cell counts per donor to identify mega-donors
donor_counts = (
    adata_filtered.obs.groupby(["disease", "donor_id"], observed=True)
    .size()
    .reset_index(name="cell_count")
)

# 3. Stratify and split the DONORS (60/20/20)
train_eval_donors, test_donors = train_test_split(
    donor_counts, test_size=0.20, stratify=donor_counts["disease"], random_state=42
)

train_donors, eval_donors = train_test_split(
    train_eval_donors,
    test_size=0.25,
    stratify=train_eval_donors["disease"],
    random_state=42,
)

train_ids = train_donors["donor_id"].tolist()
eval_ids = eval_donors["donor_id"].tolist()
test_ids = test_donors["donor_id"].tolist()

# 4. CAP MEGA-DONORS IN TRAINING TO PREVENT OVERFITTING
# Max 2000 cells per donor in the training set
MAX_CELLS_PER_DONOR = 2000
final_training_indices = []

for donor in train_ids:
    donor_indices = adata_filtered.obs[
        adata_filtered.obs["donor_id"] == donor
    ].index.tolist()
    if len(donor_indices) > MAX_CELLS_PER_DONOR:
        np.random.seed(42)
        donor_indices = np.random.choice(
            donor_indices, MAX_CELLS_PER_DONOR, replace=False
        ).tolist()
    final_training_indices.extend(donor_indices)

# Keep all validation and test cells intact for honest evaluation
val_test_indices = adata_filtered.obs[
    adata_filtered.obs["donor_id"].isin(eval_ids + test_ids)
].index.tolist()
final_keep_indices = final_training_indices + val_test_indices

# Update your anndata object before tokenization/preparation
adata_final = adata_filtered[final_keep_indices].copy()

# 5. Build your Geneformer split dictionaries
train_test_id_split_dict = {
    "attr_key": "individual",
    "train": train_ids + eval_ids,
    "test": test_ids,
}

train_valid_id_split_dict = {
    "attr_key": "individual",
    "train": train_ids,
    "valid": eval_ids,
}

cc.prepare_data(
    input_data_file=str(TOKENIZED_DIR / "ss_tokenized.dataset"),
    output_directory=str(output_dir),
    output_prefix=output_prefix,
    split_id_dict=train_test_id_split_dict,
)
# %%

all_metrics = cc.validate(
    model_directory=GENEFORMER
    / "Geneformer-V1-10M",  # set to V1 model to fit barely onto local resources -- V1 is faster
    prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    output_directory=os.path.abspath(output_dir),
    output_prefix=output_prefix,
    # split_id_dict=train_valid_id_split_dict,
    attr_to_split="individual",
    attr_to_balance=["disease", "age", "sex"],
    # attr_to_balance=["disease"],
    n_hyperopt_trials=0,  # Number of trials to run for hyperparameter optimization. Set it to 0 for direct training without hyperparameter optimization.
)

# %%
all_metrics = cc.evaluate_saved_model(
    model_directory=f"{output_dir}/{datestamp_min}_geneformer_cellClassifier_{output_prefix}/ksplit1/",
    # model_directory=best_checkpoint,
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    test_data_file=f"{output_dir}/{output_prefix}_labeled_test.dataset",
    output_directory=output_dir,
    output_prefix=output_prefix,
)

# %%
cc.plot_conf_mat(
    conf_mat_dict={"Geneformer": all_metrics["conf_matrix"]},
    output_directory=output_dir,
    output_prefix=output_prefix,
)
cc.plot_roc(
    roc_metric_dict={"Geneformer": all_metrics["all_roc_metrics"]},
    model_style_dict={"Geneformer": {"color": "red", "linestyle": "-"}},
    title="Dosage-sensitive vs -insensitive factors",
    output_directory=output_dir,
    output_prefix=output_prefix,
)
all_metrics


# %% EMBEDDING EXTRACTION

embex = EmbExtractor(
    model_type="CellClassifier",  # set to GeneClassifier or Pretrained for those model types
    num_classes=2,  # number of classes of fine-tuned model
    filter_data={
        "cell-type": [
            "Macrophage",
            "dendritic cell",
            "B cell",
            "Plasma cell",
            "CD4-positive, alpha-beta T cell",
            "CD8-positive, alpha-beta cytotoxic T cell",
            "CD8-positive, alpha-beta regulatory T cell",
            "effector CD8-positive, alpha-beta T cell",
            "mature NK T cell",
            "fibroblast",
            "endothelial cell",
            "smooth muscle cell",
            "acinar cell of salivary gland",
            "duct epithelial cell",
            "myoepithelial cell",
            "ionocyte",
        ]
    },
    max_ncells=94116,  # use full dataset
    emb_layer=0,  # extracts embeddings from last layer
    emb_label=["disease", "individual"],
    labels_to_plot=["disease", "individual"],
    forward_batch_size=128,
    nproc=8,
    model_version="V1",  # default is V2, here set to V1 model to fit into Colab 40G GPU resources
)

model_path = (
    f"{output_dir}/{datestamp_min}_geneformer_cellClassifier_{output_prefix}/ksplit1/"
)

embs = embex.extract_embs(
    model_path,
    TOKENIZED_DIR / "ss_tokenized.dataset",
    EMB_DIR,
    "ss_finetuned_embs",
)

# %%
embex.plot_embs(
    embs=embs,
    plot_style="umap",
    output_directory=EMB_DIR,
    output_prefix="emb_umap",
    max_ncells_to_plot=10000,
)

embex.plot_embs(
    embs=embs,
    plot_style="heatmap",
    output_directory=EMB_DIR,
    output_prefix="emb_heatmap",
    max_ncells_to_plot=10000,
)

# %%
