"""
Procurement Auctions (PM) - Win Prediction Code Sample

Goal:
- Predict whether a bidder wins a PM auction (win ∈ {0,1})
- Use only ex ante observables (no bid itself, no post outcomes)
- Provide econometric baseline + ML benchmark
- Evaluate with group split by auction (jpnumber) to avoid leakage

Data expectation:
- CSV at: data/raw/pm_bidder_level.csv
- One row per bidder × PM auction
- Must include:
    label: win
    group: jpnumber
    features (recommended): is_large_firm, lengest, lnumb, util, cross_type_count

This script implements a predictive (not causal) modeling pipeline using
ex ante bidder and auction characteristics. Key design choices include:

- auction-level group splits to avoid leakage across bidders,
- transparent econometric baselines,
- evaluation using both ranking (AUC) and calibration (Brier score).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay



# ---------------------------------------------------------------------
# Paths / Column names
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "pm_bidder_level.csv"

LABEL_COL = "win"
GROUP_COL = "jpnumber"


def _clean_label(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Ensure label is numeric and in {0,1}. Drop NA/inf and invalid values.
    """
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in dataset.")

    df = df.copy()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    before = len(df)
    df = df[np.isfinite(df[label_col])]
    df = df[df[label_col].isin([0, 1])].copy()
    after = len(df)

    print(f"Label cleaning: kept {after}/{before} rows (dropped {before-after}).")
    return df


def _build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build preprocessing pipeline: median impute for numeric, most-frequent + one-hot for categorical.
    """
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor, num_cols, cat_cols

# Split by auction (jpnumber) so bidders from the same auction never appear in both training and test sets.
def _group_train_test_split(X, y, groups, test_size=0.25, random_state=42):
    """
    Group split so the same auction (jpnumber) does not appear in both train and test.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def _print_cv_auc(model, X_train, y_train, n_splits=5, random_state=42, label="Model"):
    """
    5-fold CV AUC on training set only (no groups here; this is for quick stability check).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\n{label} {n_splits}-fold CV AUC (train only):")
    print(f"  mean={scores.mean():.4f}, std={scores.std():.4f}, scores={np.round(scores, 4)}")


def main():
    # -----------------------------------------------------------------
    # 1) Load data
    # -----------------------------------------------------------------
    print("Loading data...")
    df = pd.read_csv(DATA_RAW)
    print(f"Raw data shape: {df.shape}")

    # -----------------------------------------------------------------
    # 2) Clean label
    # -----------------------------------------------------------------
    df = _clean_label(df, LABEL_COL)

    # -----------------------------------------------------------------
    # 3) Check group column
    # -----------------------------------------------------------------
    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column '{GROUP_COL}' not found. "
                         f"Needed for group split to avoid leakage.")

    # -----------------------------------------------------------------
    # 4) Define feature sets (baseline vs full)
    #     - baseline: common, interpretable predictors
    #     - full: add capacity and cross-type competition structure
    # -----------------------------------------------------------------
    baseline_features = ["is_large_firm", "lengest", "lnumb"]
    full_features = ["is_large_firm", "lengest", "lnumb", "util", "cross_type_count"]

    # Keep only columns that exist (robust to naming differences)
    baseline_features = [c for c in baseline_features if c in df.columns]
    full_features = [c for c in full_features if c in df.columns]

    if len(baseline_features) < 2:
        raise ValueError("Too few baseline features found. Check column names in CSV.")

    print("\nFeature sets:")
    print(f"  Baseline: {baseline_features}")
    print(f"  Full    : {full_features}")

    y = df[LABEL_COL].astype(int)
    groups = df[GROUP_COL]

    # -----------------------------------------------------------------
    # 5) Group split (auction-level split)
    # -----------------------------------------------------------------
    # Baseline split
    Xb = df[baseline_features].copy()
    Xb_train, Xb_test, y_train, y_test = _group_train_test_split(Xb, y, groups)

    # Full split uses same indices for fairness
    Xf = df[full_features].copy()
    Xf_train = Xf.loc[Xb_train.index]
    Xf_test  = Xf.loc[Xb_test.index]

    print("\nGroup split sizes:")
    print(f"  Train: {len(y_train)}  Test: {len(y_test)}")

    # -----------------------------------------------------------------
    # 6) LOGIT baseline model
    # -----------------------------------------------------------------
    pre_b, num_b, cat_b = _build_preprocessor(Xb_train)
    logit = LogisticRegression(max_iter=4000)

    pipe_logit_b = Pipeline([("pre", pre_b), ("model", logit)])

    _print_cv_auc(pipe_logit_b, Xb_train, y_train, label="Logit (baseline features)")
    pipe_logit_b.fit(Xb_train, y_train)

    p_b = pipe_logit_b.predict_proba(Xb_test)[:, 1]
    auc_b = roc_auc_score(y_test, p_b)
    brier_b = brier_score_loss(y_test, p_b)

    print("\nLogit baseline TEST:")
    print(f"  AUC   : {auc_b:.4f}")
    print(f"  Brier : {brier_b:.4f}")

    # -----------------------------------------------------------------
    # 7) LOGIT full model
    # -----------------------------------------------------------------
    pre_f, num_f, cat_f = _build_preprocessor(Xf_train)
    pipe_logit_f = Pipeline([("pre", pre_f), ("model", LogisticRegression(max_iter=4000))])

    _print_cv_auc(pipe_logit_f, Xf_train, y_train, label="Logit (full features)")
    pipe_logit_f.fit(Xf_train, y_train)

    p_f = pipe_logit_f.predict_proba(Xf_test)[:, 1]
    auc_f = roc_auc_score(y_test, p_f)
    brier_f = brier_score_loss(y_test, p_f)

    print("\nLogit full TEST:")
    print(f"  AUC   : {auc_f:.4f}")
    print(f"  Brier : {brier_f:.4f}")
    print(f"  ΔAUC (full - baseline): {auc_f - auc_b:+.4f}")


    # For mixed types with one-hot, we'd extract names via get_feature_names_out().
    if len(cat_f) == 0:
        coefs = pipe_logit_f.named_steps["model"].coef_.ravel()
        print("\nLogit full coefficients (numeric features):")
        for name, c in zip(num_f, coefs[:len(num_f)]):
            print(f"  {name:20s} coef={c:+.4f}  OR={np.exp(c):.3f}")

    # -----------------------------------------------------------------
    # 8) Tree model (nonlinear benchmark) on FULL numeric features only
    # -----------------------------------------------------------------
    # Tree models need numeric input: use numeric subset and median imputation.
    num_only = [c for c in full_features if c in num_f]  # numeric features found
    Xn_train = Xf_train[num_only].copy()
    Xn_test  = Xf_test[num_only].copy()

    # simple numeric preprocessing: median impute
    imp = SimpleImputer(strategy="median")
    Xn_train_imp = pd.DataFrame(imp.fit_transform(Xn_train), columns=num_only, index=Xn_train.index)
    Xn_test_imp  = pd.DataFrame(imp.transform(Xn_test), columns=num_only, index=Xn_test.index)

    tree = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
    )

    # CV on training only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tree_scores = cross_val_score(tree, Xn_train_imp, y_train, cv=cv, scoring="roc_auc")

    print("\nTree (HistGB) 5-fold CV AUC (train only):")
    print(f"  mean={tree_scores.mean():.4f}, std={tree_scores.std():.4f}, scores={np.round(tree_scores, 4)}")

    tree.fit(Xn_train_imp, y_train)
    p_tree = tree.predict_proba(Xn_test_imp)[:, 1]

    auc_tree = roc_auc_score(y_test, p_tree)
    brier_tree = brier_score_loss(y_test, p_tree)

    print("\nTree (HistGB) TEST:")
    print(f"  AUC   : {auc_tree:.4f}")
    print(f"  Brier : {brier_tree:.4f}")
    print(f"  ΔAUC (tree - logit full): {auc_tree - auc_f:+.4f}")

    # -----------------------------------------------------------------
    # 9) Save metrics
    # -----------------------------------------------------------------
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "metrics.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Win prediction (PM auctions)\n\n")
        f.write(f"Data rows used after label cleaning: {len(df)}\n")
        f.write(f"Group split by: {GROUP_COL}\n\n")

        f.write("Logit baseline\n")
        f.write(f"  features: {baseline_features}\n")
        f.write(f"  test_auc: {auc_b:.4f}\n")
        f.write(f"  test_brier: {brier_b:.4f}\n\n")

        f.write("Logit full\n")
        f.write(f"  features: {full_features}\n")
        f.write(f"  test_auc: {auc_f:.4f}\n")
        f.write(f"  test_brier: {brier_f:.4f}\n")
        f.write(f"  delta_auc_vs_baseline: {auc_f - auc_b:+.4f}\n\n")

        f.write("Tree (HistGB) full numeric\n")
        f.write(f"  numeric_features: {num_only}\n")
        f.write(f"  test_auc: {auc_tree:.4f}\n")
        f.write(f"  test_brier: {brier_tree:.4f}\n")
        f.write(f"  delta_auc_vs_logit_full: {auc_tree - auc_f:+.4f}\n")

    print(f"\nSaved metrics to: {out_path}")

    # -----------------------------------------------------------------
    # 10) Plots: calibration and ROC
    # -----------------------------------------------------------------
    fig_dir = ROOT / "reports"
    fig_dir.mkdir(exist_ok=True)

    # --- Calibration curve (Logit full vs Tree) ---
    # Use uniform bins; plot predicted probability vs empirical frequency
    prob_true_logit, prob_pred_logit = calibration_curve(y_test, p_f, n_bins=10, strategy="uniform")
    prob_true_tree,  prob_pred_tree  = calibration_curve(y_test, p_tree, n_bins=10, strategy="uniform")

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(prob_pred_logit, prob_true_logit, marker="o", label="Logit (full)")
    plt.plot(prob_pred_tree,  prob_true_tree,  marker="o", label="Tree (HistGB)")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical win rate")
    plt.title("Calibration curve (test set)")
    plt.legend()
    plt.tight_layout()
    cal_path = fig_dir / "calibration_curve.png"
    plt.savefig(cal_path, dpi=200)
    plt.close()

    # --- ROC curve (Logit full vs Tree) ---
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, p_f, name="Logit (full)")
    RocCurveDisplay.from_predictions(y_test, p_tree, name="Tree (HistGB)")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC curves (test set)")
    plt.tight_layout()
    roc_path = fig_dir / "roc_curves.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()

    print(f"Saved figures to:\n  {cal_path}\n  {roc_path}")



if __name__ == "__main__":
    main()
