import pandas as pd
import importlib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sccoda.util import comp_ana as mod
from sccoda.util import cell_composition_data as dat
from sccoda.util import data_visualization as viz

import sccoda.datasets as scd
from sccoda.model.scCODA_model import EricaModel

def run_hmc_once(
    data_matrix,
    covariate_matrix,
    cell_types,
    covariate_names,
    formula,
    ref_index,
    priors,
    hmc_params
):
    model = EricaModel(
        reference_cell_type=ref_index,
        data_matrix=data_matrix,
        covariate_matrix=covariate_matrix,
        cell_types=cell_types,
        covariate_names=covariate_names,
        formula=formula,
        **priors
    )

    res = model.sample_hmc(**hmc_params)
    return res

step_sizes = [0.001, 0.002, 0.003, 0.005, 0.01]
leapfrogs  = [5, 10, 20, 30]


priors = dict(
    alpha_loc=0.0,
    alpha_sd=5.0,
    sigma_hc_scale=1.0,
    gamma_loc=0.0,
    gamma_sd=1.0,
    tau_temperature=50.0
)

num_results = 2000
num_burnin  = 500

datasets = sys.argv[1:]
for csv_path in datasets:

    print(f"Running sweep on: {csv_path}")

    df = pd.read_csv(csv_path)

    if "donor_id" in df.columns:
        df = df.drop(columns=["donor_id"])

    cell_types = [c for c in df.columns if c.startswith("CT")]
    covariates = [c for c in df.columns if c not in cell_types]

    data_matrix = df[cell_types].values.astype(float)
    covariate_matrix = df[covariates].values.astype(float)

    ref_index = cell_types.index("CT5")
    formula = "~ " + " + ".join(covariates)

    # Create output folder
    base = csv_path.split("/")[-1].replace(".csv", "")
    out_dir = f"hmc_sweep_{base}"
    os.makedirs(out_dir, exist_ok=True)

    results = []  
    for step in step_sizes:
        for L in leapfrogs:

            print(f"  step_size={step}, L={L}")

            hmc_params = dict(
                num_results=num_results,
                num_burnin=num_burnin,
                step_size=step,
                num_leapfrog_steps=L
            )

            res = run_hmc_once(
                data_matrix,
                covariate_matrix,
                cell_types,
                covariate_names=covariates,
                formula=formula,
                ref_index=ref_index,
                priors=priors,
                hmc_params=hmc_params
            )

            acc = res.sample_stats["is_accepted"].values.mean()

            results.append([step, L, acc])


    # Save results to CSV
    df_res = pd.DataFrame(results, columns=["step_size", "leapfrog_steps", "acceptance_rate"])
    csv_file = os.path.join(out_dir, "sweep_results.csv")
    df_res.to_csv(csv_file, index=False)
    print(f"Saved sweep results to: {csv_file}")
    
    pivot = df_res.pivot(index="step_size", columns="leapfrog_steps", values="acceptance_rate")

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f")
    plt.title(f"Acceptance Rate Heatmap: {base}")
    plt.tight_layout()

    heatmap_path = os.path.join(out_dir, "acceptance_heatmap.png")
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    print(f"Saved heatmap to: {heatmap_path}")
