# scVI Differential Expression Plots
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
mapped_csv = OUTPUT_DIR / "scvi_de_sjogren_vs_normal.mapped.csv"
df = pd.read_csv(mapped_csv, index_col=0)
adata = sc.read_h5ad(DATA_DIR / "salivary-scvi.h5ad")
model = scvi.model.SCVI.load(DATA_DIR / "scvi_model", adata=adata)

# %%
