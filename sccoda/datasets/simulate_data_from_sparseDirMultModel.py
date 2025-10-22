import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import patsy as pt
import numpy as np
import sys
import getopt
import logging

tfd = tfp.distributions

def main(args):
    N = 200
    K = 5
    SEED = 0 
    save_outputs_base = "./"
    try:
        opts, args = getopt.getopt(args, '', ["N=", "K=", "SEED=","save_outputs_base="])
        print(opts)
        for opt, arg in opts:
            if opt == '--N':
                N = int(arg)
            elif opt == '--K':
                K = int(arg)
            elif opt == '--SEED':
                SEED = int(arg)
            elif opt == '--save_outputs_base':
                save_outputs_base = arg
    except getopt.GetoptError: 
        sys.exit()
    # -----------------------------
    # 1. Simulation parameters
    # -----------------------------
    tf.random.set_seed(SEED)  
    P = 15  # total number of covariates 
    n_counts = tf.random.uniform((N,), minval=50, maxval=200, dtype=tf.int32) # number of trials per observation (varies per observation). Alternatively, use a fixed number like n_counts = 100
    n_counts = tf.cast(n_counts, tf.float32) #
    # -----------------------------
    # 2. Covariate matrix
    # -----------------------------
    # Continuous covariates
    X_cont = tf.random.normal((N, P - 5))

    # Categorical covariates (3 categorical columns, each with 3 levels)
    X_cat = tf.random.uniform((N, 3), minval=0, maxval=3, dtype=tf.int32) + 1
    X_cat_df = pd.DataFrame(X_cat.numpy(), columns=["cat0", "cat1", "cat2"])
    X_cat_matrix = pt.dmatrix("C(cat0) + C(cat1) + C(cat2) - 1", data=X_cat_df)
    X_cat_tf = tf.convert_to_tensor(X_cat_matrix, dtype=tf.float32)

    # Binary covariates
    X_bin = tf.random.uniform((N, 2), minval=0, maxval=2, dtype=tf.int32) + 1
    X_bin = tf.cast(X_bin, tf.float32)

    # Combine
    X = tf.concat([X_cont, X_cat_tf, X_bin], axis=1)
    # mistake bc one-hot encoding increases number of columns
    P = X.shape[1]
    assert X.shape == (N, P)

    # -----------------------------
    # 3. Sparse true coefficients
    # -----------------------------
    true_beta = tf.Variable(tf.zeros((P, K), dtype=tf.float32))
    true_beta[:2, :-1].assign(tf.random.normal((2, K - 1)))
    true_beta[-1, :-1].assign(tf.random.normal((K - 1,)))
    true_beta[-3, :-1].assign(tf.random.normal((K - 1,)))
    true_beta = tf.convert_to_tensor(true_beta)

    # -----------------------------
    # 4. Precision parameter
    # -----------------------------
    true_phi = tf.constant(15.0, dtype=tf.float32)

    # -----------------------------
    # 5. Compute Dirichlet parameters
    # -----------------------------
    eta = tf.matmul(X, true_beta)
    mu_true = tf.nn.softmax(eta, axis=-1)
    alpha_true = mu_true * true_phi

    # -----------------------------
    # 6. Sample from Dirichlet–Multinomial
    # -----------------------------
    dirichlet_multinomial = tfd.DirichletMultinomial(
        total_count=n_counts,
        concentration=alpha_true
    )
    y = dirichlet_multinomial.sample()  # shape (N, K)

    # -----------------------------
    # 7. Check results
    # -----------------------------
    print("X shape:", X.shape)
    print("alpha_true shape:", alpha_true.shape)
    print("y shape:", y.shape)

    # run different ns for tau_temperature, then other parameters
    # -----------------------------
    # 8. Save simulated data to scCODA input format
    # -----------------------------
    donor_id = [f"S{i:03d}" for i in range(1, N + 1)]
    cell_types = [f"CT{j+1}" for j in range(K)]

    cont_names = [f"x_cont{i+1}" for i in range(X_cont.shape[1])]
    cat_names = list(X_cat_matrix.design_info.column_names)
    bin_names = [f"x_bin{i+1}" for i in range(X_bin.shape[1])]

    covariates_df = pd.DataFrame(
        np.concatenate([
            X_cont.numpy(),          # (N, 10)
            X_cat_tf.numpy(),        # (N, 9)
            X_bin.numpy()            # (N, 2)
        ], axis=1),
        columns=cont_names + cat_names + bin_names
    )
    counts_df = pd.DataFrame(y.numpy().astype(int), columns=cell_types)
    final_df = pd.concat([pd.DataFrame({'donor_id': donor_id}), covariates_df, counts_df], axis=1)

    out_csv = f"{save_outputs_base}/scCODA_simulated_N={N}_K={K}_P={P}_SEED={SEED}.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"Saved scCODA CSV to: {out_csv}")



if __name__ == "__main__":
    main(sys.argv[1:])