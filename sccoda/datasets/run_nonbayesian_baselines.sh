#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 72:00:00
#SBATCH -J sbatch_wrapper_run
#SBATCH --mem=64GB
#SBATCH --partition=batch
#SBATCH -e  /users/ebrown62/scratch2/%x-log-%j.err
#SBATCH -o /users/ebrown62/scratch2/%x-log-%j.out

# Specify an output file
#SBATCH --mail-type=END,FAIL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=erica_brown3@brown.edu # Email to which notifications will be sent


CURRDIR=/users/ebrown62/scCODA/sccoda/datasets; 
cd /users/ebrown62/scCODA/sccoda/datasets;
module load python/3.11.0s-ixrhc3q;
module load r/4.4.0-yycctsj; 
module load llvm/16.0.2; 
 
module load pcre2/10.42 texlive/20220321; 
module load cmake/3.26.3; 
module load libgit2/1.6.4; 
module load geos/3.11.2; 
module load libpng/1.6.39; 
module load gdal/3.7.0 proj/9.2.0; 
module load cuda/12.2.0 cudnn/8.9.6.50 openssl libarchive/3.6.2; 
module load graphviz inkscape; 
module load hdf5; 
module load gsl;
module load julia;
module load jags;
export LD_PRELOAD=/gpfs/runtime/opt/intel/2020.2/mkl/lib/intel64/libmkl_def.so:/gpfs/runtime/opt/intel/2020.2/mkl/lib/intel64/libmkl_avx2.so:/gpfs/runtime/opt/intel/2020.2/mkl/lib/intel64/libmkl_core.so:/gpfs/runtime/opt/intel/2020.2/mkl/lib/intel64/libmkl_intel_lp64.so:/gpfs/runtime/opt/intel/2020.2/mkl/lib/intel64/libmkl_intel_thread.so:/gpfs/runtime/opt/intel/2020.2/lib/intel64_lin/libiomp5.so;


PROJECT_ROOT=/users/ebrown62/scCODA/sccoda/datasets
DATA_DIR=$PROJECT_ROOT
OUT_BASE=$PROJECT_ROOT/results/covariate_scaling  

mkdir -p "$OUT_BASE"


#Activate env
source /users/ebrown62/scCODA/sccoda-py311/bin/activate
export PYTHONPATH=/users/ebrown62/scCODA:${PYTHONPATH}

cd "$DATA_DIR"

MODELS=(
    "Haber"
    "CLR"
    "TTest"
    "CLR_ttest"
    "ALR_ttest"
    "ALR_wilcoxon"
)

cd "$DATA_DIR"
SAVE_DIR=$PROJECT_ROOT/datasets/results/covariate_scaling
for FILE in scCODA_simulated_N=200_K=5_P=*"_SEED=0.csv"; do

    if [[ "$FILE" != *true_beta* ]]; then
        echo "Processing dataset: $FILE"
        PVAL=$(echo "$FILE" | sed -E 's/.*P=([0-9]+)_SEED.*/\1/')
        for MODEL in "${MODELS[@]}"; do
            echo "  Running model: $MODEL"
            MODEL_DIR="$OUT_BASE/$MODEL"
            mkdir -p "$MODEL_DIR"
            P_DIR="$MODEL_DIR/P=${PVAL}"
            mkdir -p "$P_DIR"

            python $PROJECT_ROOT/non_bayesian_baselines.py \
                --model "$MODEL" \
                --file "$FILE" \
                --out "$P_DIR"
        done

    fi

done