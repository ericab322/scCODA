import os
import sys
import getopt
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import cell_composition_data as dat
from util import data_visualization as viz

import datasets as scd

def main(argv):
    save_file = ''
    dataset_name = ''
    try:
        opts, args = getopt.getopt(argv, '', ["save_file=", "dataset_name="])#
        print(opts)
                                
        for opt, arg in opts:
            if opt == '--save_file':
                save_file = arg
            elif opt == '--dataset_name':
                dataset_name = arg


    except getopt.GetoptError: 
        sys.exit()
        print(opts)
    datadirbase  = "/".join(save_file.split("/")[:-1])
    resultsfile = f"{datadirbase}/scCODA_{dataset_name}.csv"
    # Create scCODA dataset here
    try:
        adata = ad.read_h5ad(save_file, backed='r')
        obs_df = adata.obs.copy()
    except Exception as e:
        print(f"Could not read {save_file}: {e}")
        sys.exit(1)
    cols = ['donor_id', 'cell_type']
    for col in cols:
        if col not in obs_df.columns:
            print(f"Missing '{col}' in obs_df.")
            sys.exit(1)
    three_covariates = ['age', 'sex', 'tissue']
    covariates = [c for c in three_covariates if c in obs_df.columns]
    df_counts = (
        obs_df.groupby(['donor_id', 'cell_type'])
        .size()
        .reset_index(name='count')
        .pivot(index='donor_id', columns='cell_type', values='count')
        .fillna(0)
        .astype(int)
    )
    
    if covariates:
        cov_df = obs_df.groupby('donor_id')[covariates].first()
        df_final = cov_df.join(df_counts)
    else:
        df_final = df_counts

    # Save the final DataFrame to a CSV file
    out_dir = os.path.dirname(save_file)
    csv_path = os.path.join(out_dir, f"scCODA_{dataset_name}.csv")
    h5ad_path = os.path.join(out_dir, f"scCODA_{dataset_name}.h5ad")

    print(f"Saving scCODA CSV to: {csv_path}")
    df_final.to_csv(csv_path)
    print("CSV saved.")

if __name__ == '__main__':
    main(sys.argv[1:])