import numpy as np
import pandas as pd
import argparse

def load_inferred_matrix(csv_path, prior_label="default"):
    df = pd.read_csv(csv_path)

    # Drop duplicate first column if exists
    cov_cols = [c for c in df.columns if "covariate" in c]
    if len(cov_cols) > 1:
        if np.issubdtype(df[cov_cols[0]].dtype, np.number):
            df = df.drop(columns=cov_cols[0])
        else:
            df = df.drop(columns=cov_cols[1])
        df = df.rename(columns={cov_cols[-1]: "covariate"})
    if "prior" in df.columns:
        df = df[df["prior"] == prior_label].copy()

    # index to covariate
    df = df.set_index("covariate")
    df.index = df.index.astype(str)

    # just use means
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    cell_types = [c.replace("_mean", "") for c in mean_cols]

    mat = df[mean_cols].to_numpy(dtype=float)
    covariates = list(df.index)

    return mat, covariates, cell_types


def build_full_table(true_beta_csv,
                                 braw_csv,
                                 tau_csv,
                                 prior_label="default",
                                 eps_true=1e-8):
    # true betas
    true_df = pd.read_csv(true_beta_csv, index_col=0)
    cov_true  = list(true_df.index)    
    ct_true   = list(true_df.columns) 

    # inferred matrices
    beta_mat, cov_braw, ct_beta = load_inferred_matrix(braw_csv, prior_label)
    tau_mat,  cov_tau,  ct_tau  = load_inferred_matrix(tau_csv,  prior_label)

    # find common covariates and cell types
    common_cov = [cv for cv in cov_true if cv in cov_braw and cv in cov_tau]
    common_ct  = [ct for ct in ct_true  if ct in ct_beta and ct in ct_tau]
    true_df   = true_df.loc[common_cov, common_ct]
    true_beta = true_df.to_numpy(dtype=float)

    # select inferred matrices to common covariates and cell types
    idx_cov_braw = {cv: i for i, cv in enumerate(cov_braw)}
    idx_cov_tau  = {cv: i for i, cv in enumerate(cov_tau)}
    idx_ct_beta  = {ct: j for j, ct in enumerate(ct_beta)}
    idx_ct_tau   = {ct: j for j, ct in enumerate(ct_tau)}

    beta_sel = beta_mat[
        np.ix_([idx_cov_braw[cv] for cv in common_cov],
               [idx_ct_beta[ct] for ct in common_ct])
    ]
    tau_sel = tau_mat[
        np.ix_([idx_cov_tau[cv] for cv in common_cov],
               [idx_ct_tau[ct] for ct in common_ct])
    ]

    # build driver table
    rows = []
    for i, cov in enumerate(common_cov):
        for j, ct in enumerate(common_ct):
            tb = float(true_beta[i, j])        
            bm = float(beta_sel[i, j])        
            tm = float(tau_sel[i, j])         
            rows.append({
                "covariate": cov,
                "cell_type": ct,
                "true_beta": tb,
                "beta_mean": bm,
                "tau_mean": tm,
                "true_hit": abs(tb) > eps_true, 
            })

    return pd.DataFrame(rows)


def compute_metrics(driver_df,
                    tau_threshold=0.95):

    df = driver_df.copy()
    true_hit = df["true_hit"].values
    pred_hit = (df["tau_mean"].values >= tau_threshold)

    TP = int(np.sum(true_hit & pred_hit))
    FP = int(np.sum(~true_hit & pred_hit))
    FN = int(np.sum(true_hit & ~pred_hit))
    TN = int(np.sum(~true_hit & ~pred_hit))

    # sign accuracy 
    mask_sign = true_hit & pred_hit
    if np.sum(mask_sign) > 0:
        tb = df.loc[mask_sign, "true_beta"].values
        bm = df.loc[mask_sign, "beta_mean"].values
        sign_match = (np.sign(tb) == np.sign(bm)) 
        sign_accuracy = float(np.mean(sign_match))
    else:
        sign_accuracy = np.nan

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "sign_accuracy": sign_accuracy,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true-beta")
    ap.add_argument("--braw")
    ap.add_argument("--tau", required=True)
    ap.add_argument("--prior", default="default")
    ap.add_argument("--tau-threshold", type=float, default=0.5)
    ap.add_argument("--out", default="driver_metrics.csv")
    args = ap.parse_args()

    driver_df = build_full_table(
        true_beta_csv=args.true_beta,
        braw_csv=args.braw,
        tau_csv=args.tau,
        prior_label=args.prior
    )

    metrics = compute_metrics(driver_df, tau_threshold=args.tau_threshold)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(args.out, index=False)

    print(f"Metrics saved to {args.out}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()