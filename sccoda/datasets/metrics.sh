#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 72:00:00
#SBATCH -J sbatch_wrapper_run
#SBATCH --mem=64GB
#SBATCH --partition=batch
#SBATCH -e  /users/ebrown62/scratch2/%x-log-%j.err
#SBATCH -o /users/ebrown62/scratch2/%x-log-%j.out

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=erica_brown3@brown.edu

CURRDIR=/users/ebrown62/scCODA/sccoda/datasets
cd /users/ebrown62/scCODA/sccoda/datasets

module load python/3.11.0s-ixrhc3q
module load r/4.4.0-yycctsj
module load llvm/16.0.2
 
module load pcre2/10.42 texlive/20220321
module load cmake/3.26.3
module load libgit2/1.6.4
module load geos/3.11.2
module load libpng/1.6.39
module load gdal/3.7.0 proj/9.2.0
module load cuda/12.2.0 cudnn/8.9.6.50 openssl libarchive/3.6.2
module load graphviz inkscape
module load hdf5
module load gsl
module load julia
module load jags

export LD_PRELOAD=/gpfs/runtime/opt/intel/2020.2/mkl/lib/intel64/libmkl_def.so:/gpfs/runtime/opt/intel/2020.2/mkl/lib/intel64/libmkl_avx2.so:/gpfs/runtime/opt/intel/2020.2/mkl/lib/intel64/libmkl_core.so:/gpfs/runtime/opt/intel/2020.2/mkl/lib/intel64/libmkl_intel_lp64.so:/gpfs/runtime/opt/intel/2020.2/mkl/lib/intel64/libmkl_intel_thread.so:/gpfs/runtime/opt/intel/2020.2/lib/intel64_lin/libiomp5.so

# Activate env
source /users/ebrown62/scCODA/sccoda-py311/bin/activate


BASE_TRUE="/users/ebrown62/scCODA/sccoda/datasets"
BASE_RES="/users/ebrown62/scCODA/sccoda/datasets/results/covariate_scaling/new_tau_temp"
OUTDIR="$BASE_RES/075metrics"

mkdir -p "$OUTDIR"

P_VALUES=("1" "10" "100" "1000" "5000")
CONDS=("low" "default" "high")
TAU_THRESH=0.75

for P in "${P_VALUES[@]}"; do
    for COND in "${CONDS[@]}"; do
        
        SIM_DIR="$BASE_RES/scCODA_simulated_N=200_K=5_P=${P}_SEED=0/tau_temperature"

        BRAW="$SIM_DIR/tau_temperature_braw.csv"
        TAU="$SIM_DIR/tau_temperature_tau.csv"
        TRUE="$BASE_TRUE/scCODA_simulated_N=200_K=5_P=${P}_SEED=0_true_beta.csv"

        OUTFILE="$OUTDIR/driver_metrics_P${P}_${COND}_075.csv"
        python metrics.py \
            --true-beta "$TRUE" \
            --braw "$BRAW" \
            --tau "$TAU" \
            --prior "$COND" \
            --tau-threshold $TAU_THRESH \
            --out "$OUTFILE"

    done
done
