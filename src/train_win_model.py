"""
Predicting PM procurement auction wins (code sample).

This is a predictive exercise, not a causal analysis.
Main points:
- Use auction-level group split (jpnumber) to avoid leakage across bidders in the same auction.
- Keep a small, interpretable logit baseline, then add a couple of extra predictors.
- Report both ranking (AUC) and calibration (Brier), plus simple plots.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, RocCurveDisplay
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt


def _safe_mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _clean_win_label(df: pd.DataFrame, win_col: str = "win") -> pd.DataFrame:
    """Keep only rows where win is a clean {0,1}. Drop missing or invalid labels."""
    if win_col not in df.columns:
        raise KeyError(f"Missing required label column: {win_col}")

    s = pd.to_numeric(df[win_col], errors="coerce")
    ok = s.isin([0, 1])

    kept = int(ok.sum())
    total = len(df)

    out = df.loc[ok].copy()
    out[win_col] = s.loc[ok].astype(int)

    print(f"Label cleaning: kept {kept}/{total} rows (dropped {total - kept}).")
    return out


def _group_train_test_split(df: pd.DataFrame, group_col: str, test_size: float = 0.25, seed: int = 42):
    """Split by auction group so no auction appears in both train and test sets."""
    if group_col not in df.columns:
        raise KeyError(f"Missing required group column: {group_col}")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)

    idx = np.arange(len(df))
    groups = df[group_col].to_numpy()

    train_idx, test_idx = next(gss.split(idx, groups=groups))
    return train_idx, test_idx


def _cv_auc(model, X, y, groups, n_splits: int = 5):
    """Grouped cross-validated AUC on training data only."""
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr, va in gkf.split(X, y, groups=groups):
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[va])[:, 1]
        scores.append(roc_auc_score(y[va], p))
    return np.array(scores, dtype=float)


def main():
    # Paths
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "pm_bidder_level.csv"

    reports_dir = repo_root / "reports"
    _safe_mkdir(str(reports_dir))

    metrics_path = reports_dir / "metrics.txt"
    cal_fig_path = reports_dir / "calibration_curve.png"
    roc_fig_path = reports_dir / "roc_curves.png"

    # Load data
    print("Loading data...")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Expected input file not found:\n  {data_path}\n\n"
            "Place your dataset at data/raw/pm_bidder_level.csv (not tracked by git)."
        )

    df = pd.read_csv(data_path)
    print(f"Raw data shape: {df.shape}")

    # Label cleaning + IMPORTANT: reset index so train_idx/test_idx match .loc
    df = _clean_win_label(df, win_col="win").reset_index(drop=True)

    # Feature sets
    baseline_features = ["is_large_firm", "lengest", "lnumb"]
    full_features = ["is_large_firm", "lengest", "lnumb", "util", "cross_type_count"]

    print("\nFeature sets:")
    print(f"  Baseline: {baseline_features}")
    print(f"  Full    : {full_features}")

    # Checks
    required_cols = set(["win", "jpnumber"] + full_features)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Group split
    train_idx, test_idx = _group_train_test_split(df, group_col="jpnumber", test_size=0.25, seed=42)
    print("\nGroup split sizes:")
    print(f"  Train: {len(train_idx)}  Test: {len(test_idx)}")

    y = df["win"].to_numpy(dtype=int)

    Xb = df[baseline_features].to_numpy()
    Xf = df[full_features].to_numpy()

    y_train, y_test = y[train_idx], y[test_idx]
    Xb_train, Xb_test = Xb[train_idx], Xb[test_idx]
    Xf_train, Xf_test = Xf[train_idx], Xf[test_idx]

    # Now .loc is safe because we reset_index(drop=True)
    groups_train = df.loc[train_idx, "jpnumber"].to_numpy()

    # Models
    logit = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )

    tree = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(random_state=42)),
        ]
    )

    # Logit baseline
    scores_b = _cv_auc(logit, Xb_train, y_train, groups_train, n_splits=5)
    print("\nLogit (baseline features) 5-fold CV AUC (train only):")
    print(f"  mean={scores_b.mean():.4f}, std={scores_b.std():.4f}, scores={np.round(scores_b,4)}")

    logit.fit(Xb_train, y_train)
    p_b = logit.predict_proba(Xb_test)[:, 1]
    auc_b = roc_auc_score(y_test, p_b)
    brier_b = brier_score_loss(y_test, p_b)

    print("\nLogit baseline TEST:")
    print(f"  AUC   : {auc_b:.4f}")
    print(f"  Brier : {brier_b:.4f}")

    # Logit full
    scores_f = _cv_auc(logit, Xf_train, y_train, groups_train, n_splits=5)
    print("\nLogit (full features) 5-fold CV AUC (train only):")
    print(f"  mean={scores_f.mean():.4f}, std={scores_f.std():.4f}, scores={np.round(scores_f,4)}")

    logit.fit(Xf_train, y_train)
    p_f = logit.predict_proba(Xf_test)[:, 1]
    auc_f = roc_auc_score(y_test, p_f)
    brier_f = brier_score_loss(y_test, p_f)

    print("\nLogit full TEST:")
    print(f"  AUC   : {auc_f:.4f}")
    print(f"  Brier : {brier_f:.4f}")
    print(f"  ΔAUC (full - baseline): {auc_f - auc_b:+.4f}")

    print("\nLogit full coefficients (numeric features):")
    try:
        clf = logit.named_steps["clf"]
        coefs = clf.coef_.ravel()
        for name, coef in zip(full_features, coefs):
            oratio = float(np.exp(coef))
            print(f"  {name:<18} coef={coef:+.4f}  OR={oratio:.3f}")
    except Exception as e:
        print(f"  (Could not print coefficients: {e})")

    # Tree benchmark
    scores_t = _cv_auc(tree, Xf_train, y_train, groups_train, n_splits=5)
    print("\nTree (HistGB) 5-fold CV AUC (train only):")
    print(f"  mean={scores_t.mean():.4f}, std={scores_t.std():.4f}, scores={np.round(scores_t,4)}")

    tree.fit(Xf_train, y_train)
    p_tree = tree.predict_proba(Xf_test)[:, 1]
    auc_t = roc_auc_score(y_test, p_tree)
    brier_t = brier_score_loss(y_test, p_tree)

    print("\nTree (HistGB) TEST:")
    print(f"  AUC   : {auc_t:.4f}")
    print(f"  Brier : {brier_t:.4f}")
    print(f"  ΔAUC (tree - logit full): {auc_t - auc_f:+.4f}")

    # Save metrics
    lines = []
    lines.append("=== Win prediction (PM) ===")
    lines.append(f"Rows after label cleaning: {df.shape[0]}")
    lines.append("")
    lines.append(f"Train size: {len(train_idx)}   Test size: {len(test_idx)}")
    lines.append("")
    lines.append("Logit baseline:")
    lines.append(f"  CV AUC mean={scores_b.mean():.4f}, std={scores_b.std():.4f}")
    lines.append(f"  Test AUC={auc_b:.4f}, Brier={brier_b:.4f}")
    lines.append("")
    lines.append("Logit full:")
    lines.append(f"  CV AUC mean={scores_f.mean():.4f}, std={scores_f.std():.4f}")
    lines.append(f"  Test AUC={auc_f:.4f}, Brier={brier_f:.4f}")
    lines.append(f"  ΔAUC(full-baseline)={auc_f - auc_b:+.4f}")
    lines.append("")
    lines.append("Tree (HistGB) on full features:")
    lines.append(f"  CV AUC mean={scores_t.mean():.4f}, std={scores_t.std():.4f}")
    lines.append(f"  Test AUC={auc_t:.4f}, Brier={brier_t:.4f}")
    lines.append(f"  ΔAUC(tree-logit_full)={auc_t - auc_f:+.4f}")
    lines.append("")

    metrics_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved metrics to: {metrics_path}")

    # Plots
    prob_true_logit, prob_pred_logit = calibration_curve(y_test, p_f, n_bins=10, strategy="uniform")
    prob_true_tree, prob_pred_tree = calibration_curve(y_test, p_tree, n_bins=10, strategy="uniform")

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(prob_pred_logit, prob_true_logit, marker="o", label="Logit (full)")
    plt.plot(prob_pred_tree, prob_true_tree, marker="o", label="Tree (HistGB)")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical win rate")
    plt.title("Calibration curve (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cal_fig_path, dpi=200)
    plt.close()

    plt.figure()
    RocCurveDisplay.from_predictions(y_test, p_f, name="Logit (full)")
    RocCurveDisplay.from_predictions(y_test, p_tree, name="Tree (HistGB)")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC curves (test set)")
    plt.tight_layout()
    plt.savefig(roc_fig_path, dpi=200)
    plt.close()

    print("Saved figures to:")
    print(f"  {cal_fig_path}")
    print(f"  {roc_fig_path}")


if __name__ == "__main__":
    main()
