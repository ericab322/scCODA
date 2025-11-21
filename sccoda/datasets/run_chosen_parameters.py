# Setup
import importlib
import warnings
warnings.filterwarnings("ignore")

import os
import re
import argparse


import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from sccoda.util import cell_composition_data as dat
from sccoda.util import data_visualization as viz
import numpy as np

import sccoda.datasets as scd

import time
import tracemalloc

from sccoda.model.scCODA_model import EricaModel

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

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
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

def save_inferred_values(res, prior_label="default"):
    b_raw_da = res.posterior["b_raw"]
    tau_da   = res.posterior["ind"] 
    reduce_dims = [d for d in b_raw_da.dims if d in ("chain", "draw")]

    b_raw_mean = b_raw_da.mean(dim=reduce_dims)
    b_raw_sd   = b_raw_da.std(dim=reduce_dims)
    
    tau_mean= tau_da.mean(dim=reduce_dims)
    tau_sd  = tau_da.std(dim=reduce_dims)

    covariates = list(b_raw_mean.coords["covariate"].values)
    cell_types = list(b_raw_mean.coords["cell_type_nb"].values)
    
    data_dict = {}
    for j, ct in enumerate(cell_types):
        data_dict[f"{ct}_braw_mean"] = np.asarray(b_raw_mean)[:, j]
        data_dict[f"{ct}_braw_sd"]   = np.asarray(b_raw_sd)[:, j]
        data_dict[f"{ct}_tau_mean"]  = np.asarray(tau_mean)[:, j]
        data_dict[f"{ct}_tau_sd"]    = np.asarray(tau_sd)[:, j]

    def build_df(mean_da, sd_da, label):
        data_dict = {}
        for j, ct in enumerate(cell_types):
            data_dict[f"{ct}_mean"] = np.asarray(mean_da)[:, j]
            data_dict[f"{ct}_sd"]   = np.asarray(sd_da)[:, j]
        df = pd.DataFrame(data_dict, index=covariates).reset_index()
        df.rename(columns={"index": "covariate"}, inplace=True)
        df.insert(1, "prior", label)
        return df

    braw_df = build_df(b_raw_mean, b_raw_sd, prior_label)
    tau_df  = build_df(tau_mean, tau_sd, prior_label)

    return braw_df, tau_df

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

def run_hmc(
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

    tracemalloc.start()
    start = time.perf_counter()
    res = model.sample_hmc(**hmc_params)
    end = time.perf_counter()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return res, dict(
        elapsed=end - start,
        current_MB=cur / 1e6,
        peak_MB=peak / 1e6
    )

def run_default_if_needed(
    out_dir,
    data_matrix,
    covariate_matrix,
    cell_types,
    covariate_names,
    formula,
    ref_index,
    default_priors,
    hmc_params,
    force=False
):
    default_path = os.path.join(out_dir, "summary_default.csv")

    if os.path.exists(default_path) and not force:
        print("[default] Cached results found — skipping.")
        return pd.read_csv(default_path)

    print("[default] Running default model...")
    res, prof = run_hmc(
        data_matrix,
        covariate_matrix,
        cell_types,
        covariate_names,
        formula,
        ref_index,
        default_priors,
        hmc_params
    )

    df_default = summarize(res, "default", reference=ref_index)
    df_default.to_csv(default_path, index=False)
    return df_default

def run_sweep_for_params(
    sweep_params,
    sweep_specs,
    default_priors,
    out_dir,
    data_matrix,
    covariate_matrix,
    cell_types,
    covariate_names,
    formula,
    ref_index,
    hmc_params
):
    # We re-run *default* once so we have b_raw and tau for default.
    print("Recomputing default state (summary + braw/tau)...")

    # full default run
    default_res, _ = run_hmc(
        data_matrix, covariate_matrix, cell_types, covariate_names,
        formula, ref_index, default_priors, hmc_params
    )
    # summaries
    df_default = summarize(default_res, "default", reference=ref_index)
    df_default.to_csv(os.path.join(out_dir, "summary_default.csv"), index=False)

    #  braw/tau values
    default_braw, default_tau = save_inferred_values(default_res, "default")

    for param in sweep_params:

        if param not in sweep_specs:
            print(f"WARNING: Unknown parameter '{param}' — skipping.")
            continue

        lvls = sweep_specs[param]
        print(f"Sweeping parameter: {param}")

        pdir = os.path.join(out_dir, param)
        os.makedirs(pdir, exist_ok=True)

        # -------------------------------------------------------
        # LOW
        # -------------------------------------------------------
        priors_low = {**default_priors, param: lvls["low"]}
        res_low, _ = run_hmc(
            data_matrix, covariate_matrix, cell_types, covariate_names,
            formula, ref_index, priors_low, hmc_params
        )
        df_low = summarize(res_low, "low", reference=ref_index)
        low_braw, low_tau = save_inferred_values(res_low, "low")

        # -------------------------------------------------------
        # HIGH
        # -------------------------------------------------------
        priors_high = {**default_priors, param: lvls["high"]}
        res_high, _ = run_hmc(
            data_matrix, covariate_matrix, cell_types, covariate_names,
            formula, ref_index, priors_high, hmc_params
        )
        df_high = summarize(res_high, "high", reference=ref_index)
        high_braw, high_tau = save_inferred_values(res_high, "high")

        all_braw = pd.concat([default_braw, low_braw, high_braw], ignore_index=True)
        all_braw.to_csv(os.path.join(pdir, f"{param}_braw.csv"), index=False)
        all_tau = pd.concat([default_tau, low_tau, high_tau], ignore_index=True)
        all_tau.to_csv(os.path.join(pdir, f"{param}_tau.csv"), index=False)

        df_all = pd.concat([df_default, df_low, df_high], ignore_index=True)
        df_all.to_csv(os.path.join(pdir, f"summary_{param}.csv"), index=False)

        fig, _ = plot_grid(df_all, param_name=param)
        fig.savefig(os.path.join(pdir, f"{param}.png"), dpi=200)
        plt.close()


def run_one_file(
    csv_path,
    out_root,
    sweep_params=None,   
    default_priors=None,
    hmc_params=None,
    force_default=False
):
    sweep_params = sweep_params or []
    default_priors = default_priors or {
        "alpha_loc": 0.0,
        "alpha_sd": 5.0,
        "sigma_hc_scale": 1.0,
        "gamma_loc": 0.0,
        "gamma_sd": 1.0,
        "tau_temperature": 50.0
    }
    hmc_params = hmc_params or dict(
        num_results=5000,
        num_burnin=1000,
        step_size=0.01,
        num_leapfrog_steps=20
    )

    data = pd.read_csv(csv_path)
    if "donor_id" in data.columns:
        data = data.set_index("donor_id")

    cell_types = [c for c in data.columns if c.startswith("CT")]
    covariates = [c for c in data.columns if c not in cell_types]

    ref_cell = "CT5"
    ref_index = cell_types.index(ref_cell)

    data_matrix = data[cell_types].values
    covariate_matrix = data[covariates].values if covariates else None
    formula = "~ " + " + ".join(covariates) if covariates else "~ 1"

    # --- output dir ---
    base = os.path.basename(csv_path).replace(".csv", "")
    out_dir = os.path.join(out_root, base)
    os.makedirs(out_dir, exist_ok=True)

    # --- run default (cached) ---
    df_default = run_default_if_needed(
        out_dir,
        data_matrix,
        covariate_matrix,
        cell_types,
        covariate_names=covariates,
        formula=formula,
        ref_index=ref_index,
        default_priors=default_priors,
        hmc_params=hmc_params,
        force=force_default
    )

    sweep_specs = {
        "alpha_loc":        dict(low=-10.0,  high=10.0),
        "alpha_sd":         dict(low=0.5,    high=10.0),
        "sigma_hc_scale":   dict(low=0.25,   high=5.0),
        "gamma_loc":        dict(low=-10.0,  high=10.0),
        "gamma_sd":         dict(low=0.25,   high=5.0),
        "tau_temperature":  dict(low=1.0,    high=100.0)
    }

    run_sweep_for_params(
        sweep_params=sweep_params,
        sweep_specs=sweep_specs,
        default_priors=default_priors,
        out_dir=out_dir,
        data_matrix=data_matrix,
        covariate_matrix=covariate_matrix,
        cell_types=cell_types,
        covariate_names=covariates,
        formula=formula,
        ref_index=ref_index,
        hmc_params=hmc_params
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", default="./results")
    ap.add_argument("--sweep", nargs="*", default=[])
    ap.add_argument("--force_default", action="store_true")
    args = ap.parse_args()

    run_one_file(
        csv_path=args.file,
        out_root=args.out,
        sweep_params=args.sweep,
        force_default=args.force_default
    )
