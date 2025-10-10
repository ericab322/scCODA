import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd

tfd = tfp.distributions

# -----------------------------
# 1. Simulation parameters
# -----------------------------
tf.random.set_seed(0)
N, P, K = 200, 15, 5  # samples, covariates, categories        
n_counts = tf.random.uniform((N,), minval=50, maxval=200, dtype=tf.int32) # number of trials per observation (varies per observation). Alternatively, use a fixed number like n_counts = 100
n_counts = tf.cast(n_counts, tf.float32) #


# -----------------------------
# 2. Covariate matrix
# -----------------------------
# Continuous covariates
X_cont = tf.random.normal((N, P - 5))

# Categorical covariates (3 categorical columns, each with 3 levels)
X_cat = tf.random.uniform((N, 3), minval=0, maxval=3, dtype=tf.int32) + 1
X_cat_df = pd.get_dummies(pd.DataFrame(X_cat.numpy()))
X_cat_tf = tf.convert_to_tensor(X_cat_df.values, dtype=tf.float32)

# Binary covariates
X_bin = tf.random.uniform((N, 2), minval=0, maxval=2, dtype=tf.int32) + 1
X_bin = tf.cast(X_bin, tf.float32)

# Combine
X = tf.concat([X_cont, X_cat_tf, X_bin], axis=1)
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


# -----------------------------
# 8. Save simulated data to scCODA input format
# -----------------------------

# Do here!