"""
03_merge_and_build_table4.py

Final assembly step: combines the outputs of all three benchmark
pipelines --
  R/03_bglr_pooled_grouped.R          -> 6 genuine BGLR models, grouped CV
  R/02_..._pooled_ordinary.R          -> 6 genuine BGLR models, ordinary CV
  python/01_benchmark_13_conventional_ml.py -> 13 sklearn/boosting models
  python/02_benchmark_10_deep_learning.py   -> 10 deep-learning models

into the final, complete 30-model (29 genuine + BayesR proxy) datasets
that feed the manuscript's Table 4, Table S5, and Table S5b.

BayesR is not computed by any of the three pipelines -- it is retained
throughout the manuscript as a disclosed regularized-regression proxy,
since BGLR has no native BayesR implementation. This script does not
fabricate a BayesR row; it is added back in manually from the original
manuscript's proxy result if you need it in the output table (see the
BAYESR_PROXY_VALUES dict below -- fill in or remove as appropriate for
your own data).

USAGE
    python3 03_merge_and_build_table4.py

Expects, in ./output/ (adjust INPUT_DIR below if different):
    genuine_pooled_summary.csv          (from 01_benchmark_13_conventional_ml.py)
    genuine_dl10_pooled_summary.csv     (from 02_benchmark_10_deep_learning.py)
    BGLR_Pooled_ordinary_summary.csv    (from R/02_..._pooled_ordinary.R)
    BGLR_Pooled_grouped_summary.csv     (from R/03_bglr_pooled_grouped.R)

Produces:
    table4_grouped_final.csv   -- the primary, leakage-safe Table 4 / Table S5
    table_S5b_leakage_delta.csv -- ordinary vs. grouped comparison, Table S5b
"""

import os
import pandas as pd

INPUT_DIR = "./output"
OUTPUT_DIR = "./output"

CLASSES = {
    'GBLUP': 'Conventional', 'rrBLUP': 'Conventional', 'BayesA': 'Conventional', 'BayesB': 'Conventional',
    'BayesC': 'Conventional', 'BayesCpi': 'Conventional', 'BayesR': 'Conventional', 'BRR': 'Conventional',
    'BL': 'Conventional', 'RKHS': 'Conventional',
    'ElasticNet': 'ML', 'RandomForest': 'ML', 'LightGBM': 'ML', 'SVR': 'ML', 'GBM': 'ML', 'XGBoost': 'ML',
    'KRR': 'ML', 'DecisionTree': 'ML', 'PLS': 'ML', 'KNN': 'ML',
    'ResNet': 'DL', 'DNN': 'DL', 'DualCNN': 'DL', 'CNN': 'DL', 'Autoencoder': 'DL', 'RNN': 'DL', 'LSTM': 'DL',
    'GRU': 'DL', 'Transformer': 'DL', 'MLP': 'DL',
}

# BayesR is never computed here (no genuine BGLR implementation exists for
# it). If you want it included as a disclosed proxy row in the output
# table, fill this in from your own proxy run; otherwise leave empty and
# it will be added as "proxy -- not evaluated".
BAYESR_PROXY_VALUES = {}  # e.g. {'Fold5': 0.524, 'SD': 0.274, 'Fold10': 0.565, 'Rep3Mean': 0.530}


def load_and_harmonize():
    win = pd.read_csv(os.path.join(INPUT_DIR, "genuine_pooled_summary.csv"))
    dl = pd.read_csv(os.path.join(INPUT_DIR, "genuine_dl10_pooled_summary.csv"))
    dl.columns = ['Scheme', 'Model', 'mean', 'std']

    bglr_ord = pd.read_csv(os.path.join(INPUT_DIR, "BGLR_Pooled_ordinary_summary.csv"))
    bglr_ord.columns = ['Scheme', 'Model', 'mean', 'std']
    bglr_grp = pd.read_csv(os.path.join(INPUT_DIR, "BGLR_Pooled_grouped_summary.csv"))
    bglr_grp.columns = ['Scheme', 'Model', 'mean', 'std']

    master = pd.concat([win, dl, bglr_ord, bglr_grp], ignore_index=True)
    # harmonize BGLR's "pooled_5fold"/"pooled_10fold" naming to match the
    # sklearn/DL pipelines' "pooled_random_5fold"/"pooled_random_10fold"
    master['Scheme'] = master['Scheme'].replace({
        'pooled_5fold': 'pooled_random_5fold',
        'pooled_10fold': 'pooled_random_10fold',
        'pooled_5fold_rep1': 'pooled_random_5fold_rep1',
        'pooled_5fold_rep2': 'pooled_random_5fold_rep2',
        'pooled_5fold_rep3': 'pooled_random_5fold_rep3',
    })
    return master


def build_table4(master):
    g5 = master[master.Scheme == 'pooled_grouped_5fold'].set_index('Model')[['mean', 'std']]
    g10 = master[master.Scheme == 'pooled_grouped_10fold'].set_index('Model')['mean']
    greps = master[master.Scheme.isin(
        ['pooled_grouped_5fold_rep1', 'pooled_grouped_5fold_rep2', 'pooled_grouped_5fold_rep3'])]
    grep_mean = greps.groupby('Model')['mean'].mean()

    rows = []
    for model, cls in CLASSES.items():
        if model == 'BayesR':
            if BAYESR_PROXY_VALUES:
                rows.append([model, cls, BAYESR_PROXY_VALUES.get('Fold5'), BAYESR_PROXY_VALUES.get('SD'),
                             BAYESR_PROXY_VALUES.get('Fold10'), BAYESR_PROXY_VALUES.get('Rep3Mean')])
            else:
                rows.append([model, cls, None, None, None, None])
            continue
        rows.append([model, cls, g5.loc[model, 'mean'], g5.loc[model, 'std'],
                     g10.loc[model], grep_mean.loc[model]])

    df = pd.DataFrame(rows, columns=['Model', 'Class', 'GroupedCV_5fold', 'SD', 'GroupedCV_10fold', 'GroupedCV_5fold_RepMean'])
    return df.sort_values('GroupedCV_5fold', ascending=False, na_position='last')


def build_leakage_delta(master):
    rand5 = master[master.Scheme == 'pooled_random_5fold'].set_index('Model')['mean']
    grp5 = master[master.Scheme == 'pooled_grouped_5fold'].set_index('Model')['mean']
    delta = (rand5 - grp5).rename('leakage_delta').sort_values(ascending=False)
    out = pd.DataFrame({
        'Model': delta.index,
        'Random_5fold': rand5.loc[delta.index].values,
        'Grouped_5fold': grp5.loc[delta.index].values,
        'leakage_delta_random_minus_grouped': delta.values,
    })
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master = load_and_harmonize()

    table4 = build_table4(master)
    table4.to_csv(os.path.join(OUTPUT_DIR, "table4_grouped_final.csv"), index=False)
    print("=== Table 4 (genotype-grouped, leakage-safe) ===")
    print(table4.round(4).to_string(index=False))

    delta = build_leakage_delta(master)
    delta.to_csv(os.path.join(OUTPUT_DIR, "table_S5b_leakage_delta.csv"), index=False)
    print("\n=== Table S5b (leakage-delta comparison) ===")
    print(delta.round(4).to_string(index=False))
    print(f"\nMean delta: {delta.leakage_delta_random_minus_grouped.mean():.4f}, "
          f"median: {delta.leakage_delta_random_minus_grouped.median():.4f}")
    print(f"Models where grouped > random (delta < 0): "
          f"{(delta.leakage_delta_random_minus_grouped < 0).sum()} of {len(delta)}")

    print(f"\nWritten to {OUTPUT_DIR}/table4_grouped_final.csv and table_S5b_leakage_delta.csv")


if __name__ == "__main__":
    main()
