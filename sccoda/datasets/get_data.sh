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

#Activate env
source /users/ebrown62/scCODA/sccoda-py311/bin/activate; 

LINK_TO_DOWNLOAD=( "https://datasets.cellxgene.cziscience.com/e11a76f0-57ff-4358-8a91-008655475059.h5ad" )
DATA_NAME=( "aging_atlas" )
SAVE_DIR=/users/ebrown62/scCODA/sccoda/datasets
for l in ${!LINK_TO_DOWNLOAD[@]};do
    LINK=${LINK_TO_DOWNLOAD[$l]}
    DNAME=${DATA_NAME[$l]}
    cd $SAVE_DIR
    wget -O adata.h5ad $LINK
    SAVE_FILE="$SAVE_DIR/adata.h5ad"
    python generate_sccoda_input.py  --save_file "$SAVE_FILE" --dataset_name "$DNAME"
done