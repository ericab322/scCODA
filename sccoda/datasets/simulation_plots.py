# Setup
import importlib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from sccoda.util import cell_composition_data as dat
from sccoda.util import data_visualization as viz
import numpy as np

import sccoda.datasets as scd

from sccoda.model.scCODA_model import EricaModel

# Functions

def values(arr):
        mean = arr.mean(axis=(0,1))               
        lo = np.percentile(arr, 2.5, axis=(0,1))  
        hi = np.percentile(arr, 97.5, axis=(0,1))  
        return mean, lo, hi
def summarize(data, prior_label, reference=0):
    "Extract summaries from inference data object"
    
    pred = data.posterior_predictive["prediction"].values
    cell_types = list(data.posterior_predictive.cell_type.values)
    reference_name = cell_types[reference]
    
    # Overall expected samples
    pred_mean, pred_lo, pred_hi = values(pred)            
    pred_mean = pred_mean.mean(axis=0)                   
    pred_lo   = pred_lo.mean(axis=0)
    pred_hi   = pred_hi.mean(axis=0)
    
     # intercept expected sample
    alpha = data.posterior["alpha"].values  
    alpha_mean, alpha_lo, alpha_hi = values(alpha)  
    
     # effect expected sample
    beta = data.posterior["beta"].values[..., 0, :]
    beta_mean, beta_lo, beta_hi = values(beta)
    
    #log fold change
    ref_pred = pred[..., reference:reference+1]
    logfc = np.log2(pred / ref_pred)                    
    logfc_mean, logfc_lo, logfc_hi = values(logfc)       
    logfc_mean = logfc_mean.mean(axis=0)           
    logfc_lo   = logfc_lo.mean(axis=0)
    logfc_hi   = logfc_hi.mean(axis=0)
    
    rows = []
    for i, ct in enumerate(cell_types):
        rows.append({
            "cell_type": ct, "reference": reference_name, "prior": prior_label,
            "metric": "expected_sample",
            "mean": pred_mean[i], "lo": pred_lo[i], "hi": pred_hi[i]
        })
        rows.append({
            "cell_type": ct, "reference": reference_name, "prior": prior_label,
            "metric": "intercept_expected_sample",
            "mean": alpha_mean[i], "lo": alpha_lo[i], "hi": alpha_hi[i]
        })
        rows.append({
            "cell_type": ct, "reference": reference_name, "prior": prior_label,
            "metric": "effect_expected_sample",
            "mean": beta_mean[i], "lo": beta_lo[i], "hi": beta_hi[i]
        })
        rows.append({
            "cell_type": ct, "reference": reference_name, "prior": prior_label,
            "metric": "log2FC",
            "mean": logfc_mean[i], "lo": logfc_lo[i], "hi": logfc_hi[i]
        })

    return pd.DataFrame(rows)

import seaborn as sns
import matplotlib.pyplot as plt

def plot_grid(df, param_name=None):
    metrics = ["expected_sample", "log2FC", "effect_expected_sample", "intercept_expected_sample"]
    titles  = ["Expected Samples", "Log2 Fold Change", "Effect Expected Sample", "Intercept Expected Sample"]

    n_metrics = len(metrics)
    cell_types = df["cell_type"].unique()
    n_cells = len(cell_types)

    prior_order = ["low", "default", "high"]

    palette = {
        "expected_sample": "#4C72B0",         
        "log2FC": "#55A868",                   
        "effect_expected_sample": "#C44E52",   
        "intercept_expected_sample": "#8172B2" 
    }

    fig, axes = plt.subplots(n_cells, n_metrics, figsize=(4*n_metrics, 3*n_cells), sharex=False)

    if n_cells == 1:
        axes = axes[np.newaxis, :]  

    for i, ct in enumerate(cell_types):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            sub = df[(df["cell_type"] == ct) & (df["metric"] == metric)].copy()

            # sort priors
            sub["prior"] = pd.Categorical(sub["prior"], categories=prior_order, ordered=True)
            sub = sub.sort_values("prior")

            color = palette[metric]

            if metric == "expected_sample":
                ax.bar(sub["prior"], sub["mean"], color=color, alpha=0.8)
            else:
                ax.plot(sub["prior"], sub["mean"], marker="o", color=color, linewidth=2)
                ax.fill_between(
                    sub["prior"], sub["lo"], sub["hi"],
                    color=color, alpha=0.2
                )

            ax.set_xticks(range(len(prior_order)))
            ax.set_xticklabels(prior_order, rotation=45, ha="right", fontsize=9)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)

            if i == 0:
                ax.set_title(titles[j], fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(ct, fontsize=9, fontweight="bold")
            else:
                ax.set_ylabel("")

    if param_name is not None:
        fig.suptitle(f"{param_name}", fontsize=14, fontweight="bold", y=1.02)
        
    plt.tight_layout()
    return fig, axes

def select_reference_cell_type(data: pd.DataFrame, threshold: float = 0.05):
    columns = data.columns.tolist()
    cell_types = [col for col in columns if col.startswith("CT")]

    percent_zero = np.sum(data[cell_types] == 0, axis=0) / len(data)
    nonrare_ct = np.where(percent_zero < threshold)[0]

    row_totals = data[cell_types].sum(axis=1).replace(0, np.nan)
    rel_abun = data[cell_types].div(row_totals, axis=0)
    ra_vals = rel_abun.to_numpy()

    cell_type_disp = np.var(ra_vals, axis=0) / np.mean(ra_vals, axis=0)

    min_var_idx_within = np.argmin(cell_type_disp[nonrare_ct])
    ref_index = nonrare_ct[min_var_idx_within]
    ref_cell_type = cell_types[ref_index]
    return ref_index, ref_cell_type

# Main Script
import os
import re
import argparse

def run_one_file(csv_path: str, out_root: str,
                 num_results=5000, num_burnin=1000, step_size=0.01, num_leapfrog_steps=20):
    
    base = os.path.basename(csv_path)
    m = re.search(r"n=(\d+)", base)
    n_tag = f"n={m.group(1)}" if m else "n=unknown"
    out_dir = os.path.join(out_root, n_tag)
    os.makedirs(out_dir, exist_ok=True)
    
    data = pd.read_csv(csv_path)
    if "donor_id" in data.columns:
        data = data.set_index("donor_id")
    
    columns = data.columns.tolist()
    cell_types = [c for c in columns if c.startswith("CT")]
    covariates = [c for c in columns if c not in cell_types]
    ref_index, ref_cell_type = select_reference_cell_type(data, threshold=0.05)
    
    data_matrix = data[cell_types].to_numpy(dtype=float)
    covariate_matrix = data[covariates].to_numpy(dtype=float) if len(covariates) else None
    covariate_names = covariates
    formula = "~ " + " + ".join(covariates) if len(covariates) else "~ 1"
    
    # HMC
    hmc_kwargs = dict(
        num_results=num_results,
        num_burnin=num_burnin,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
    )

    # default
    base_model = EricaModel(
        reference_cell_type=ref_index,
        data_matrix=data_matrix,
        covariate_matrix=covariate_matrix,
        cell_types=cell_types,
        covariate_names=covariate_names,
        formula=formula
    )
    base_res = base_model.sample_hmc(**hmc_kwargs)

    df_default = summarize(base_res, prior_label="default", reference=ref_index)
    df_default.to_csv(os.path.join(out_dir, "summary_default.csv"), index=False)
    
    # sweep over priors
    sweep_specs = {
        "alpha_loc":        dict(low=0.5,  high=10.0),
        "alpha_sd":         dict(low=0.5,  high=10.0),
        "sigma_hc_scale":   dict(low=0.25, high=5.0),
        "gamma_loc":        dict(low=0.25, high=5.0),
        "gamma_sd":         dict(low=0.25, high=5.0),
        "tau_temperature":  dict(low=1.0,  high=100.0),
    }
    default_priors = dict(
        alpha_loc=0.0,
        alpha_sd=5.0,
        sigma_hc_scale=1.0,
        gamma_loc=0.0,
        gamma_sd=1.0,
        tau_temperature=50.0,
    )

    for param, lvls in sweep_specs.items():
        print(f"[{n_tag}] Sweeping prior: {param}")
        pdir = os.path.join(out_dir, param)
        os.makedirs(pdir, exist_ok=True)

        # low
        priors_low = {**default_priors, param: lvls["low"]}
        model_low = EricaModel(
            reference_cell_type=ref_index,
            data_matrix=data_matrix,
            covariate_matrix=covariate_matrix,
            cell_types=cell_types,
            covariate_names=covariate_names,
            formula=formula,
            **priors_low
        )
        res_low = model_low.sample_hmc(**hmc_kwargs)

        # high
        priors_high = {**default_priors, param: lvls["high"]}
        model_high = EricaModel(
            reference_cell_type=ref_index,
            data_matrix=data_matrix,
            covariate_matrix=covariate_matrix,
            cell_types=cell_types,
            covariate_names=covariate_names,
            formula=formula,
            **priors_high
        )
        res_high = model_high.sample_hmc(**hmc_kwargs)
        
        # summarize and plot
        df_low  = summarize(res_low,  prior_label="low",  reference=ref_index)
        df_high = summarize(res_high, prior_label="high", reference=ref_index)
        df_all  = pd.concat([df_default, df_low, df_high], ignore_index=True)
        df_all.to_csv(os.path.join(pdir, f"summary_{param}.csv"), index=False)

        fig, _ = plot_grid(df_all, param_name=param)
        fig.savefig(os.path.join(pdir, f"{param}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"[{n_tag}] Done. Results in {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to simulated CSV (e.g., datasets/scCODA_simulated_n=2.csv)")
    ap.add_argument("--out", default="./results", help="Output directory for results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_one_file(
        csv_path=args.file,
        out_root=args.out
    )