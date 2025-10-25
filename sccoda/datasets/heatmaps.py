import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# input
TRUE_BETA_PATH = "/users/ebrown62/scCODA/sccoda/datasets/scCODA_simulated_N=2_K=5_P=16_SEED=0_true_beta.csv"
RESULTS_DIR = "/users/ebrown62/scCODA/sccoda/datasets/results/n=unknown"
REF_BASELINE = "CT5"      
TAU_THRESHOLD = 0.5
VMIN, VMAX = -1, 1

true_beta = pd.read_csv(TRUE_BETA_PATH)
if true_beta.columns[0].startswith("Unnamed"):
    true_beta = true_beta.set_index(true_beta.columns[0])
true_beta = true_beta[[c for c in true_beta.columns if c.startswith("CT") and c != REF_BASELINE]]

# true beta heatmap
plt.figure(figsize=(10, 4))
plt.imshow(true_beta.to_numpy(dtype=float), aspect="auto", cmap="bwr", vmin=VMIN, vmax=VMAX)
plt.colorbar(label="True beta")
plt.xlabel("Dirichlet categories")
plt.ylabel("Covariates")
plt.title(f"True beta coefficients ({REF_BASELINE} dropped)")
plt.tight_layout()
out_true = os.path.join(RESULTS_DIR, "plots")
os.makedirs(out_true, exist_ok=True)
plt.savefig(os.path.join(out_true, "true_beta.png"), dpi=200)
plt.close()
print(f"Saved true_beta.png → {out_true}")

param_folders = [d for d in os.listdir(RESULTS_DIR)
                 if os.path.isdir(os.path.join(RESULTS_DIR, d)) and not d.startswith("plots")]

for param in param_folders:
    param_dir = os.path.join(RESULTS_DIR, param)
    braw_path = os.path.join(param_dir, f"{param}_braw.csv")
    tau_path  = os.path.join(param_dir, f"{param}_tau.csv")

    if not (os.path.exists(braw_path) and os.path.exists(tau_path)):
        print(f"Skipping {param}: missing files.")
        continue

    out_dir = os.path.join(param_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # load
    b_raw = pd.read_csv(braw_path)
    tau   = pd.read_csv(tau_path)
    priors = sorted(b_raw["prior"].unique())

    for prior in priors:
        print(f"Plotting {param} — {prior}")

        b_sub = b_raw[b_raw["prior"] == prior]
        tau_sub = tau[tau["prior"] == prior]

        b_mean = b_sub[[c for c in b_sub.columns if "mean" in c]]
        tau_mean = tau_sub[[c for c in tau_sub.columns if "mean" in c]]

        # \adjusted effect size heatmap 
        full_effect = b_mean.values * tau_mean.values
        plt.figure(figsize=(10, 4))
        plt.imshow(full_effect, aspect="auto", cmap="bwr", vmin=VMIN, vmax=VMAX)
        plt.colorbar(label="Posterior mean effects")
        plt.xlabel("Dirichlet categories")
        plt.ylabel("Covariates")
        plt.title(f"{param} ({prior}) — Posterior mean of beta coefficients")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{param}_{prior}_effects.png"), dpi=200)
        plt.close()

        # greater than threshold heatmap 
        tau_mask = tau_mean.where(tau_mean > TAU_THRESHOLD, np.nan)
        plt.figure(figsize=(10, 4))
        plt.imshow(tau_mask.to_numpy(dtype=float),
                   aspect="auto", cmap="Reds", vmin=0, vmax=1)
        plt.colorbar(label=f"τ (only > {TAU_THRESHOLD} shown)")
        plt.xlabel("Dirichlet categories")
        plt.ylabel("Covariates")
        plt.title(f"{param} ({prior}) — τ > {TAU_THRESHOLD}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{param}_{prior}_tau_mask.png"), dpi=200)
        plt.close()

print("All plots completed.")
