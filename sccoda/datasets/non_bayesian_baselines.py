import importlib
import warnings
warnings.filterwarnings("ignore")

import os
import re
import argparse


import pandas as pd
from anndata import AnnData
import pickle as pkl
import matplotlib.pyplot as plt
from sccoda.util import cell_composition_data as dat
from sccoda.util import data_visualization as viz
import numpy as np

import sccoda.datasets as scd

import time
import tracemalloc

from sccoda.model.other_models import (
    SimpleModel,
    HaberModel, CLRModel, TTest, CLRModel_ttest,
    ALRModel_ttest, ALRModel_wilcoxon,
    ALDEx2Model, DirichRegModel,
    BetaBinomialModel, ANCOMBCModel
)

def make_anndata(data_df, covariates, cell_types):
    ad = AnnData(
        X=data_df[cell_types].values,
        obs=data_df[covariates],
        var=pd.DataFrame(index=cell_types)
    )
    return ad

def run_simple_model(ad, ref_index, cov_name):
    model = SimpleModel(
        reference_cell_type=ref_index,
        data_matrix=ad.X,
        covariate_matrix=ad.obs[[cov_name]].values,
        cell_types=list(ad.var.index),
        covariate_names=[cov_name],
        formula=f"~ {cov_name}"
    )
    tracemalloc.start()
    t0 = time.perf_counter()
    res = model.sample_hmc(
        num_results=6000, num_burnin=2000,
        step_size=0.01, num_leapfrog_steps=20
    )
    t1 = time.perf_counter()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return res, dict(time=t1-t0, mem=peak/1e6)

def extract_simplemodel_results(res):
    beta = res.posterior["beta"].values
    mean = beta.mean(axis=(0,1))
    return mean

def run_nonbayes_model(model_name, ad, reference_idx, covariate):

    if model_name == "Haber":
        m = HaberModel(ad, covariate_column=covariate)
        m.fit_model()
        return m.p_val

    if model_name == "CLR":
        m = CLRModel(ad, covariate_column=covariate)
        m.fit_model()
        return m.p_val

    if model_name == "TTest":
        m = TTest(ad, covariate_column=covariate)
        m.fit_model()
        return m.p_val

    if model_name == "CLR_ttest":
        m = CLRModel_ttest(ad, covariate_column=covariate)
        m.fit_model()
        return m.p_val

    if model_name == "ALR_ttest":
        m = ALRModel_ttest(ad, covariate_column=covariate)
        m.fit_model(reference_cell_type=reference_idx)
        return m.p_val

    if model_name == "ALR_wilcoxon":
        m = ALRModel_wilcoxon(ad, covariate_column=covariate)
        m.fit_model(reference_cell_type=reference_idx)
        return m.p_val
    
    # r models

    if model_name == "ALDEx2":
        m = ALDEx2Model(ad, covariate_column=covariate)
        m.fit_model()
        return m.p_val

    if model_name == "DirichReg":
        m = DirichRegModel(ad, covariate_column=covariate)
        m.fit_model()
        return m.p_val

    if model_name == "BetaBinomial":
        m = BetaBinomialModel(ad, covariate_column=covariate)
        m.fit_model()
        return m.p_val

    if model_name == "ANCOMBC":
        m = ANCOMBCModel(ad, covariate_column=covariate)
        m.fit_model()
        return m.p_val
def run_nonbayes(model_name, csv_path, out_dir):

    df = pd.read_csv(csv_path)
    if "donor_id" in df.columns:
        df = df.set_index("donor_id")

    celltypes = [c for c in df.columns if c.startswith("CT")]
    covs = [c for c in df.columns if c.startswith("C(cat0)[1]")]
    cov = covs[0] 
    ref_idx = celltypes.index("CT5")

    ad = make_anndata(df, [cov], celltypes)

    os.makedirs(out_dir, exist_ok=True)

    pvals = run_nonbayes_model(model_name, ad, ref_idx, cov)
    out_path = os.path.join(out_dir, f"{model_name}.csv")
    pd.DataFrame([pvals], columns=celltypes).assign(model=model_name).to_csv(out_path, index=False)

    print(f"[{model_name}] saved to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_nonbayes(args.model, args.file, args.out)