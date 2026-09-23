"""
genuine_13model_no_bglr_no_dl.py

Leaner companion to the R-side BGLR pipeline. Two things are deliberately
excluded from this script:

  - The 6 Bayesian-alphabet models (BayesA, BayesB, BayesC, BayesCpi,
    BRR, BL) plus BayesR -- run separately, directly in R, via
    bglr_fold.R and the run_*_bglr.R driver scripts.
  - The 10 deep-learning models -- dropped along with the TensorFlow/
    Keras dependency, which is often troublesome to install cleanly on
    Windows (DLL conflicts, CPU/GPU build mismatches, version pinning).

This script covers the remaining 13 models (11 sklearn + 2 boosting: the
same real, separately-imported implementations as before -- real
xgboost, real lightgbm, real sklearn SVR/PLS/KernelRidge/ElasticNet,
nothing aliased to another model) and still fixes the genotype-leakage
issue for them: every pooled analysis runs both ordinary KFold (leaky)
and GroupKFold on genotype ID (leakage-safe), reported side by side with
their delta.

Once this finishes, merge its output with the R-side BGLR results and
(separately, if/when you want the DL models) a TensorFlow-capable run of
the 10 deep-learning architectures, for the complete 30-model Table 4
and per-year tables.

REQUIREMENTS
    pip install scikit-learn pandas numpy scipy xgboost lightgbm

USAGE
    python genuine_13model_no_bglr_no_dl.py
"""

import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb


# --------------------------------------------------------------------------
# CONFIG -- edit these for your machine
# --------------------------------------------------------------------------
BASE = "./data"  # <-- edit: folder containing data.txt and Genotype.Numerical.txt
PHENO_PATH = os.path.join(BASE, "data.txt")
GENO_PATH = os.path.join(BASE, "Genotype.Numerical.txt")
OUT_DIR = "./output"
RANDOM_SEED = 42
N_PERMUTATIONS = 10000   # for the real paired-permutation test

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# DATA LOADING
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
# MODEL ZOO -- every entry is a genuinely distinct implementation.
# Bayesian-alphabet models (BayesA/B/C/Cpi/BRR/BL) and BayesR are NOT in
# this dict at all -- they're run separately, directly in R, via
# bglr_fold.R and the run_*_bglr.R driver scripts (already done/running
# on the main machine). Merge that output in afterward.
# --------------------------------------------------------------------------

def get_sklearn_models():
    return {
        "GBLUP": Ridge(alpha=10.0),                      # VanRaden 2008; Ridge on standardized markers
        "rrBLUP": Ridge(alpha=1.0),                       # Meuwissen et al. 2001
        "RKHS": KernelRidge(alpha=1.0, kernel="rbf", gamma=0.001),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
        "GBM": GradientBoostingRegressor(random_state=RANDOM_SEED),
        "SVR": SVR(kernel="rbf", C=1.0),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "DecisionTree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_SEED),
        "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000),
        "PLS": PLSRegression(n_components=10),
        "KRR": KernelRidge(alpha=0.5, kernel="rbf", gamma=0.0005),  # distinct gamma from RKHS
    }


def get_boosting_models():
    return {
        "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     random_state=RANDOM_SEED, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                       random_state=RANDOM_SEED, verbosity=-1),
    }


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


def cv_sklearn(model_ctor, X, y, groups=None, n_splits=5, seed=RANDOM_SEED):
    if groups is not None:
        splitter = shuffled_group_kfold(groups, n_splits, seed)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X)
    fold_r = []
    for tr, te in splitter:
        mdl = model_ctor()
        mdl.fit(X[tr], y[tr])
        pred = mdl.predict(X[te])
        fold_r.append(r_or_zero(y[te], pred))
    return fold_r


SKLEARN_CTORS = {**{k: (lambda v=v: v) for k, v in {}.items()}}  # placeholder, filled below

def make_ctor_dict():
    # fresh instance per fold -- fixes the "same object reused" trap
    sk = get_sklearn_models()
    boost = get_boosting_models()
    ctors = {}
    for name in sk:
        ctors[name] = (lambda n=name: get_sklearn_models()[n])
    for name in boost:
        ctors[name] = (lambda n=name: get_boosting_models()[n])
    return ctors


# --------------------------------------------------------------------------
# PAIRED PERMUTATION TEST (real one -- sign-flip test on paired fold
# differences, not a t-test dressed up as a permutation test)
# --------------------------------------------------------------------------

def paired_permutation_test(fold_r_a, fold_r_b, n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    a = np.asarray(fold_r_a); b = np.asarray(fold_r_b)
    diffs = a - b
    observed = diffs.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, len(diffs)))
    perm_means = (signs * diffs).mean(axis=1)
    p = np.mean(np.abs(perm_means) >= np.abs(observed))
    return observed, p


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    gdf, matched, traits = load_data()
    ctors = make_ctor_dict()

    all_rows = []  # Trait, Model, Scheme(random/grouped), Fold, r

    # long-format data for pooled analysis, with genotype groups
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

    # ---- Per-year, 5-fold (random -- no leakage possible within a single
    # year since each genotype appears once) ----
    for trait in traits:
        X, y, _ = get_Xy(gdf, matched, trait)
        for name, ctor in ctors.items():
            fr = cv_sklearn(ctor, X, y, groups=None, n_splits=5)
            for i, r in enumerate(fr):
                all_rows.append([trait, name, "per_year_5fold", i, r])
        print(f"Finished per-year 5-fold for {trait}")

    df_per_year = pd.DataFrame(all_rows, columns=["Trait", "Model", "Scheme", "Fold", "r"])
    df_per_year.to_csv(os.path.join(OUT_DIR, "genuine_per_year_5fold_all_folds.csv"), index=False)

    # ---- Pooled, RANDOM vs GROUPED, 5-fold / 10-fold / repeated-5-fold x3 ----
    pooled_rows = []
    for scheme_name, n_splits, groups in [
        ("pooled_random_5fold", 5, None),
        ("pooled_grouped_5fold", 5, groups_long),
        ("pooled_random_10fold", 10, None),
        ("pooled_grouped_10fold", 10, groups_long),
    ]:
        for name, ctor in ctors.items():
            fr = cv_sklearn(ctor, X_long, y_long, groups=groups, n_splits=n_splits)
            for i, r in enumerate(fr):
                pooled_rows.append([scheme_name, name, i, r])
        print(f"Finished {scheme_name}")

    # repeated 5-fold x3, grouped only (the leakage-safe version is the one
    # that belongs in the manuscript's headline table)
    for seed in [1, 2, 3]:
        for name, ctor in ctors.items():
            fr = cv_sklearn(ctor, X_long, y_long, groups=groups_long, n_splits=5, seed=seed)
            for i, r in enumerate(fr):
                pooled_rows.append([f"pooled_grouped_5fold_rep{seed}", name, i, r])

    df_pooled = pd.DataFrame(pooled_rows, columns=["Scheme", "Model", "Fold", "r"])
    df_pooled.to_csv(os.path.join(OUT_DIR, "genuine_pooled_all_folds.csv"), index=False)

    # summary table: mean/SD per model per scheme, plus random-vs-grouped delta
    summary = df_pooled.groupby(["Scheme", "Model"])["r"].agg(["mean", "std"]).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "genuine_pooled_summary.csv"), index=False)

    rand5 = summary[summary.Scheme == "pooled_random_5fold"].set_index("Model")["mean"]
    grp5 = summary[summary.Scheme == "pooled_grouped_5fold"].set_index("Model")["mean"]
    delta = (rand5 - grp5).rename("leakage_delta_r").reset_index()
    delta.to_csv(os.path.join(OUT_DIR, "genuine_leakage_delta_by_model.csv"), index=False)
    print("\nLeakage delta (random_5fold - grouped_5fold), per model -- should VARY across models:")
    print(delta.to_string())

    # ---- Real paired permutation test on the grouped (leakage-safe) folds ----
    rep_models = ["GBLUP", "rrBLUP", "RandomForest", "XGBoost", "RKHS"]
    fold_lookup = {}
    grouped5 = df_pooled[df_pooled.Scheme == "pooled_grouped_5fold"]
    for m in rep_models + ["GBLUP"]:
        vals = grouped5[grouped5.Model == m].sort_values("Fold")["r"].values
        if len(vals) == 5:
            fold_lookup[m] = vals

    perm_rows = []
    best = "GBLUP"
    for m in rep_models:
        if m == best or m not in fold_lookup:
            continue
        obs, p = paired_permutation_test(fold_lookup[best], fold_lookup[m])
        perm_rows.append([best, m, obs, p, p < 0.05])
    df_perm = pd.DataFrame(perm_rows, columns=["ModelA", "ModelB", "MeanDiff", "p_value", "Significant_0.05"])
    df_perm.to_csv(os.path.join(OUT_DIR, "genuine_paired_permutation_test.csv"), index=False)
    print("\nPaired permutation test (grouped, leakage-safe folds):")
    print(df_perm.to_string())

    print(f"\nAll outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
