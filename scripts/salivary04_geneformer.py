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