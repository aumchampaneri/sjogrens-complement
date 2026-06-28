# scVI Differential Expression
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
import torch

# %%
adata = sc.read_h5ad(f"{DATA_DIR}/salivary-scvi.h5ad")
# %%
