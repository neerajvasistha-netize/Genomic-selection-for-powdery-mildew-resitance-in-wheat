"""
genuine_10dl_models_windows.py

For the Windows laptop, running in parallel with the Windows laptop's
genuine_13model_no_bglr_no_dl.py (11 sklearn + 2 boosting models) and
the main machine's R/BGLR pipeline (6 Bayesian-alphabet + BayesR).

This script covers ONLY the 10 deep-learning models -- DNN, MLP, CNN,
DualCNN, ResNet, RNN, LSTM, GRU, Transformer, Autoencoder -- each a
real, architecturally distinct Keras model (not one MLP with ten random
seeds). Same fix as the other two pipelines: every pooled analysis runs
both ordinary KFold (leaky) and GroupKFold on genotype ID (leakage-safe),
reported side by side with their delta.

Once all three pipelines finish, their outputs merge into the complete
30-model Table 4 and per-year tables:
  - Windows:    genuine_13model_no_bglr_no_dl output (11 sklearn + 2 boosting)
  - This script: genuine_dl10 output (10 DL models)
  - Main Mac:   BGLR_PerTrait_*, BGLR_Pooled_* (6 Bayesian + BayesR proxy)

REQUIREMENTS
    pip install tensorflow pandas numpy scipy scikit-learn

RUNTIME WARNING
    10 DL models x 5 folds x (3 per-year + 4 pooled schemes + 3 reps = 10
    scheme-equivalents) = up to 500 individual Keras fits. Even small,
    on CPU, this adds up -- budget real time. EPOCHS/PATIENCE below are
    set conservatively (60/8, not 200/15) specifically so this finishes
    in a realistic window; raise them later for a final, unhurried run
    if you want to squeeze out more accuracy.

USAGE
    python3 genuine_10dl_models_windows.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress noisy TF/oneDNN/Grappler log spam

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras import layers, models as kmodels

# --------------------------------------------------------------------------
# CONFIG -- EDIT THIS PATH for the MacBook Air's actual data location
# --------------------------------------------------------------------------
BASE = "./data"  # <-- edit: folder containing data.txt and Genotype.Numerical.txt
PHENO_PATH = os.path.join(BASE, "data.txt")
GENO_PATH = os.path.join(BASE, "Genotype.Numerical.txt")
OUT_DIR = "./output"
RANDOM_SEED = 42
EPOCHS = 60          # was 200 -- lowered so a full run finishes in a realistic time
PATIENCE = 8         # was 15 -- early stopping patience, scaled down with EPOCHS

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# DATA LOADING (identical to the other two pipelines, for consistency)
# --------------------------------------------------------------------------

def load_data():
    with open(GENO_PATH) as f:
        first = f.readline().strip()
    try:
        int(first)
        geno = pd.read_csv(GENO_PATH, sep=r'\s+', skiprows=1, engine='python')
    except ValueError:
        geno = pd.read_csv(GENO_PATH, sep=r'\s+', engine='python')
    if str(geno.columns[0]).isdigit():
        geno = geno.rename(columns={geno.columns[0]: "IID"})

    ids = geno.iloc[:, 0].astype(str).values
    X = geno.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    keep = X.std(axis=0) > 1e-6
    X = X[:, keep]
    Xs = StandardScaler().fit_transform(X)
    gdf = pd.DataFrame(Xs, index=ids)

    pheno = pd.read_csv(PHENO_PATH, sep=None, engine='python')
    pheno.columns = [str(c).strip() for c in pheno.columns]
    pheno['GID'] = pheno.iloc[:, 0].astype(str)
    traits = [c for c in pheno.columns if 'DS' in c.upper()][:3]

    matched = pheno[pheno['GID'].isin(gdf.index)]
    if len(matched) == 0:
        pheno['GID2'] = pd.to_numeric(pheno['GID'], errors='coerce').astype('Int64').astype(str)
        gdf.index = pd.to_numeric(gdf.index, errors='coerce').astype('Int64').astype(str)
        matched = pheno[pheno['GID2'].isin(gdf.index)].copy()
        matched['GID'] = matched['GID2']

    print(f"Loaded {gdf.shape[0]} genotypes x {gdf.shape[1]} SNPs after "
          f"monomorphic filtering; {len(matched)} phenotyped lines matched.")
    return gdf, matched, traits


def get_Xy(gdf, matched, trait):
    df_t = matched.dropna(subset=[trait])
    valid = [g for g in df_t['GID'] if g in gdf.index]
    X = np.vstack([gdf.loc[g].values for g in valid])
    y = pd.to_numeric(df_t.set_index('GID').loc[valid, trait], errors='coerce').values
    mask = ~np.isnan(y)
    return X[mask], y[mask], np.array(valid)[mask]


# --------------------------------------------------------------------------
# DEEP LEARNING -- ten architecturally distinct Keras models
# --------------------------------------------------------------------------

def build_dnn(n_features):
    m = kmodels.Sequential([
        layers.Input((n_features,)),
        layers.Dense(256, activation="relu"), layers.Dropout(0.3),
        layers.Dense(128, activation="relu"), layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def build_mlp(n_features):
    m = kmodels.Sequential([
        layers.Input((n_features,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def build_cnn(n_features):
    m = kmodels.Sequential([
        layers.Input((n_features, 1)),
        layers.Conv1D(32, 15, strides=4, activation="relu"),
        layers.Conv1D(16, 7, strides=2, activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def build_dualcnn(n_features):
    inp = layers.Input((n_features, 1))
    b1 = layers.Conv1D(16, 5, strides=2, activation="relu")(inp)
    b1 = layers.GlobalAveragePooling1D()(b1)
    b2 = layers.Conv1D(16, 25, strides=6, activation="relu")(inp)
    b2 = layers.GlobalAveragePooling1D()(b2)
    x = layers.Concatenate()([b1, b2])
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = kmodels.Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    return m

def build_resnet(n_features):
    inp = layers.Input((n_features,))
    x = layers.Dense(64, activation="relu")(inp)
    skip = x
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64)(x)
    x = layers.Add()([x, skip])
    x = layers.ReLU()(x)
    out = layers.Dense(1)(x)
    m = kmodels.Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    return m

def build_rnn(n_features):
    # Same fix as build_transformer: a raw recurrent layer unrolling
    # sequentially over ~20,996 timesteps is computationally infeasible on
    # CPU (confirmed: ~46x slower than the downsampled version at pooled
    # data scale). Downsample first with a strided conv, then recur over
    # the much shorter reduced sequence.
    stride = max(1, n_features // 150)
    inp = layers.Input((n_features, 1))
    x = layers.Conv1D(8, kernel_size=stride, strides=stride, activation="relu")(inp)
    x = layers.SimpleRNN(32)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = kmodels.Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    return m

def build_lstm(n_features):
    stride = max(1, n_features // 150)
    inp = layers.Input((n_features, 1))
    x = layers.Conv1D(8, kernel_size=stride, strides=stride, activation="relu")(inp)
    x = layers.LSTM(32)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = kmodels.Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    return m

def build_gru(n_features):
    stride = max(1, n_features // 150)
    inp = layers.Input((n_features, 1))
    x = layers.Conv1D(8, kernel_size=stride, strides=stride, activation="relu")(inp)
    x = layers.GRU(32)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = kmodels.Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    return m

def build_transformer(n_features):
    inp = layers.Input((n_features, 1))
    # Attention cost is O(seq_len^2) -- running it directly over ~20,000 SNP
    # positions blows up memory (a [batch,heads,20996,20996] matrix is tens
    # of GB). Downsample the sequence length first with a strided conv,
    # THEN attend over the much shorter reduced sequence -- same idea as
    # the CNN/DualCNN builders' downsampling. Stride/kernel are computed
    # from n_features (targeting ~150 reduced positions) instead of a
    # fixed constant, so this works regardless of marker count.
    stride = max(1, n_features // 150)
    x = layers.Conv1D(16, kernel_size=stride, strides=stride, activation="relu")(inp)
    x = layers.Dense(16)(x)
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=8)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = kmodels.Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    return m

def build_autoencoder(n_features):
    inp = layers.Input((n_features,))
    enc = layers.Dense(128, activation="relu")(inp)
    bottleneck = layers.Dense(16, activation="relu")(enc)
    dec = layers.Dense(128, activation="relu")(bottleneck)
    recon = layers.Dense(n_features, name="recon")(dec)
    pred = layers.Dense(1, name="pred")(bottleneck)
    m = kmodels.Model(inp, [recon, pred])
    m.compile(optimizer="adam", loss={"recon": "mse", "pred": "mse"},
              loss_weights={"recon": 0.3, "pred": 1.0})
    return m

DL_BUILDERS = {
    "DNN": build_dnn, "MLP": build_mlp, "CNN": build_cnn, "DualCNN": build_dualcnn,
    "ResNet": build_resnet, "RNN": build_rnn, "LSTM": build_lstm, "GRU": build_gru,
    "Transformer": build_transformer, "Autoencoder": build_autoencoder,
}
SEQUENCE_MODELS = {"CNN", "DualCNN", "ResNet", "RNN", "LSTM", "GRU", "Transformer"}


def fit_predict_dl(name, X_tr, y_tr, X_te):
    tf.random.set_seed(RANDOM_SEED)
    n_features = X_tr.shape[1]
    builder = DL_BUILDERS[name]
    model = builder(n_features)

    Xtr_in, Xte_in = X_tr, X_te
    if name in SEQUENCE_MODELS and name not in ("ResNet",):
        Xtr_in = X_tr[..., None]
        Xte_in = X_te[..., None]

    es = tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)

    if name == "Autoencoder":
        model.fit(Xtr_in, {"recon": Xtr_in, "pred": y_tr},
                  epochs=EPOCHS, batch_size=16, verbose=0, callbacks=[es],
                  validation_split=0.15)
        _, pred = model.predict(Xte_in, verbose=0)
        return pred.ravel()

    model.fit(Xtr_in, y_tr, epochs=EPOCHS, batch_size=16, verbose=0,
              callbacks=[es], validation_split=0.15)
    pred = model.predict(Xte_in, verbose=0)
    return pred.ravel()


# --------------------------------------------------------------------------
# CV RUNNERS
# --------------------------------------------------------------------------

def r_or_zero(y_true, y_pred):
    y_pred = np.asarray(y_pred).ravel()
    if np.std(y_pred) < 1e-8 or np.std(y_true) < 1e-8:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return 0.0 if np.isnan(r) else float(r)


def shuffled_group_kfold(groups, n_splits, seed):
    """GroupKFold has no random_state/shuffle option in scikit-learn -- its
    fold assignment is deterministic from group order alone, so calling it
    with different seeds silently produces IDENTICAL folds every time.
    This shuffles the unique group order first (with the given seed), then
    assigns folds round-robin -- same no-genotype-crosses-folds guarantee
    as GroupKFold, but genuinely different splits per seed."""
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    fold_of_group = {g: i % n_splits for i, g in enumerate(unique_groups)}
    fold_ids = np.array([fold_of_group[g] for g in groups])
    for k in range(n_splits):
        te = np.where(fold_ids == k)[0]
        tr = np.where(fold_ids != k)[0]
        yield tr, te


def cv_dl(name, X, y, groups=None, n_splits=5, seed=RANDOM_SEED):
    if groups is not None:
        splitter = shuffled_group_kfold(groups, n_splits, seed)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X)
    fold_r = []
    for tr, te in splitter:
        pred = fit_predict_dl(name, X[tr], y[tr], X[te])
        fold_r.append(r_or_zero(y[te], pred))
    return fold_r


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    gdf, matched, traits = load_data()

    per_year_partial_path = os.path.join(OUT_DIR, "genuine_dl10_per_year_5fold_all_folds_partial.csv")
    pooled_partial_path = os.path.join(OUT_DIR, "genuine_dl10_pooled_all_folds_partial.csv")

    if os.path.exists(per_year_partial_path):
        all_rows = pd.read_csv(per_year_partial_path).values.tolist()
        done_per_year = set(
            (t, m) for (t, m), g in pd.read_csv(per_year_partial_path).groupby(["Trait", "Model"])
            if len(g) == 5
        )
        print(f"Resuming: found {len(done_per_year)} already-completed (trait, model) pairs "
              f"in {per_year_partial_path}")
    else:
        all_rows = []
        done_per_year = set()

    if os.path.exists(pooled_partial_path):
        pooled_rows = pd.read_csv(pooled_partial_path).values.tolist()
        pooled_df_existing = pd.read_csv(pooled_partial_path)
        done_pooled = set()
        for (s, m), g in pooled_df_existing.groupby(["Scheme", "Model"]):
            expected_folds = 10 if "10fold" in s else 5
            if len(g) == expected_folds:
                done_pooled.add((s, m))
        print(f"Resuming: found {len(done_pooled)} already-completed (scheme, model) pairs "
              f"in {pooled_partial_path}")
    else:
        pooled_rows = []
        done_pooled = set()

    long_rows = []
    for _, row in matched.iterrows():
        for yr in traits:
            v = pd.to_numeric(row[yr], errors='coerce')
            if not np.isnan(v) and row['GID'] in gdf.index:
                long_rows.append((row['GID'], yr, v))
    long_df = pd.DataFrame(long_rows, columns=["GID", "Year", "y"])
    X_long = np.vstack([gdf.loc[g].values for g in long_df['GID']])
    y_long = long_df['y'].values
    groups_long = long_df['GID'].values

    # ---- Per-year, 5-fold ----
    for trait in traits:
        X, y, _ = get_Xy(gdf, matched, trait)
        for name in DL_BUILDERS:
            if (trait, name) in done_per_year:
                print(f"{trait} {name} already done -- skipping")
                continue
            fr = cv_dl(name, X, y, groups=None, n_splits=5)
            for i, r in enumerate(fr):
                all_rows.append([trait, name, "per_year_5fold", i, r])
            print(f"{trait} {name} done, mean r = {np.mean(fr):.4f}")
            # incremental save after every model, so an interruption doesn't lose everything
            pd.DataFrame(all_rows, columns=["Trait", "Model", "Scheme", "Fold", "r"]).to_csv(
                per_year_partial_path, index=False)
        print(f"Finished per-year 5-fold for {trait}")

    df_per_year = pd.DataFrame(all_rows, columns=["Trait", "Model", "Scheme", "Fold", "r"])
    df_per_year.to_csv(os.path.join(OUT_DIR, "genuine_dl10_per_year_5fold_all_folds.csv"), index=False)

    # ---- Pooled, RANDOM vs GROUPED, 5-fold / 10-fold / repeated-5-fold x3 ----
    for scheme_name, n_splits, groups in [
        ("pooled_random_5fold", 5, None),
        ("pooled_grouped_5fold", 5, groups_long),
        ("pooled_random_10fold", 10, None),
        ("pooled_grouped_10fold", 10, groups_long),
    ]:
        for name in DL_BUILDERS:
            if (scheme_name, name) in done_pooled:
                print(f"{scheme_name} {name} already done -- skipping")
                continue
            fr = cv_dl(name, X_long, y_long, groups=groups, n_splits=n_splits)
            for i, r in enumerate(fr):
                pooled_rows.append([scheme_name, name, i, r])
            print(f"{scheme_name} {name} done, mean r = {np.mean(fr):.4f}")
            # incremental save after every model
            pd.DataFrame(pooled_rows, columns=["Scheme", "Model", "Fold", "r"]).to_csv(
                pooled_partial_path, index=False)
        print(f"Finished {scheme_name}")

    for seed in [1, 2, 3]:
        scheme_name = f"pooled_grouped_5fold_rep{seed}"
        for name in DL_BUILDERS:
            if (scheme_name, name) in done_pooled:
                print(f"{scheme_name} {name} already done -- skipping")
                continue
            fr = cv_dl(name, X_long, y_long, groups=groups_long, n_splits=5, seed=seed)
            for i, r in enumerate(fr):
                pooled_rows.append([scheme_name, name, i, r])
            pd.DataFrame(pooled_rows, columns=["Scheme", "Model", "Fold", "r"]).to_csv(
                pooled_partial_path, index=False)
        print(f"Finished {scheme_name}")

    df_pooled = pd.DataFrame(pooled_rows, columns=["Scheme", "Model", "Fold", "r"])
    df_pooled.to_csv(os.path.join(OUT_DIR, "genuine_dl10_pooled_all_folds.csv"), index=False)

    summary = df_pooled.groupby(["Scheme", "Model"])["r"].agg(["mean", "std"]).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "genuine_dl10_pooled_summary.csv"), index=False)

    rand5 = summary[summary.Scheme == "pooled_random_5fold"].set_index("Model")["mean"]
    grp5 = summary[summary.Scheme == "pooled_grouped_5fold"].set_index("Model")["mean"]
    delta = (rand5 - grp5).rename("leakage_delta_r").reset_index()
    delta.to_csv(os.path.join(OUT_DIR, "genuine_dl10_leakage_delta_by_model.csv"), index=False)
    print("\nLeakage delta (random_5fold - grouped_5fold), per DL model:")
    print(delta.to_string())

    print(f"\nAll DL outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
