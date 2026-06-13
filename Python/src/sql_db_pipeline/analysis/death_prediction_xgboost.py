#!/usr/bin/env python3
"""
Death Prediction (XGBoost + SHAP)
=================================
Predicts fly status (alive / dying / dead) from sliding-window behavioral features.

Uses table features_sliding_window: one row per (fly, window end date), features from
the previous 24 h [9am, 9am), label = status on the window end date. Train/val/test
split is by fly to avoid leakage. Target is rule-based: Alive→alive, Unhealthy→dying,
Dead→dead.

Usage:
  python death_prediction_xgboost.py --experiment-id N --output-dir analysis_results/death_prediction
  python death_prediction_xgboost.py --sliding-window-csv path/to/window.csv --output-dir ...
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from importlib import import_module
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for config
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config import DB_CONFIG, DATABASE_URL, USE_DATABASE
    from sqlalchemy import create_engine
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    USE_DATABASE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Feature columns in features_sliding_window (numeric only; exclude IDs and labels)
SLIDING_WINDOW_FEATURE_COLS = [
    'total_activity', 'activity_mean', 'activity_var', 'longest_zero_hours',
    'total_sleep_min', 'total_bouts', 'mean_bout_min', 'max_bout_min',
    'frag_bouts_per_hour', 'amplitude_24h',
]
NON_FEATURE_COLS = ['fly_id', 'experiment_id', 'window_end_date', 'exp_day', 'status', 'status_raw', 'days_until_death']
CLASS_ORDER = ['alive', 'dying', 'dead']
NEAR_DEATH_LABEL = 'near_death'
FAR_DEATH_LABEL = 'far'


def load_sliding_window_from_db(experiment_id=None):
    """Load sliding-window data from database by experiment_id."""
    if not USE_DATABASE or not DB_AVAILABLE:
        raise RuntimeError("Database is required. Configure config and database.")
    sys.path.insert(0, os.path.dirname(script_dir))
    step1 = import_module('1-prepare_data_and_health')
    if experiment_id is None:
        experiment_id = step1.get_latest_experiment_id()
        if experiment_id is None:
            raise ValueError("No experiment found. Specify --experiment-id or load from CSV.")
        print(f"[Load] Using latest experiment_id: {experiment_id}")
    else:
        print(f"[Load] Loading experiment_id: {experiment_id}")
    engine = create_engine(DATABASE_URL)
    query = f"""
        SELECT * FROM features_sliding_window
        WHERE experiment_id = {int(experiment_id)}
    """
    df = pd.read_sql(query, engine)
    engine.dispose()
    if df is None or len(df) == 0:
        raise ValueError(f"No sliding-window data for experiment_id={experiment_id}. Run step 3 with --build-sliding-window.")
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    print(f"✓ Loaded {len(df)} rows, {df['fly_id'].nunique()} flies")
    return df


def load_sliding_window_from_csv(path):
    """Load sliding-window data from CSV. Expects fly_id, feature columns, status."""
    df = pd.read_csv(path)
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    for c in ['fly_id', 'status']:
        if c not in df.columns:
            raise ValueError(f"CSV must contain column '{c}'.")
    print(f"✓ Loaded {len(df)} rows from {path}, {df['fly_id'].nunique()} flies")
    return df


def prepare_X_y(df, exclude_longest_zero=False, scale=False):
    """
    Build feature matrix X and target y. Filter to alive/dying/dead.
    Optionally exclude longest_zero_hours to reduce pre-bias.
    """
    feature_cols = [c for c in SLIDING_WINDOW_FEATURE_COLS if c in df.columns]
    if exclude_longest_zero and 'longest_zero_hours' in feature_cols:
        feature_cols = [c for c in feature_cols if c != 'longest_zero_hours']
    if not feature_cols:
        raise ValueError("No feature columns found.")
    # Filter to valid status
    df = df[df['status'].isin(CLASS_ORDER)].copy()
    if len(df) == 0:
        raise ValueError("No rows with status in alive/dying/dead.")
    X = df[feature_cols].copy()
    y = df['status'].copy()
    # Drop rows with NaN in features or target
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask].copy()
    y = y[mask].copy()
    df = df.loc[mask].copy()
    X.attrs['fly_id'] = df['fly_id'].values
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)
        X.attrs['scaler'] = scaler
    X.attrs['feature_cols'] = feature_cols
    return X, y, feature_cols


def prepare_X_y_from_days_until_death(df, near_death_days=2, exclude_longest_zero=False):
    """
    Build X and binary target from days_until_death: near_death (1) if 0 <= days_until_death <= near_death_days, else far (0).
    Drops rows with null days_until_death (censored or no death).
    """
    feature_cols = [c for c in SLIDING_WINDOW_FEATURE_COLS if c in df.columns]
    if exclude_longest_zero and 'longest_zero_hours' in feature_cols:
        feature_cols = [c for c in feature_cols if c != 'longest_zero_hours']
    if not feature_cols:
        raise ValueError("No feature columns found.")
    if 'days_until_death' not in df.columns:
        raise ValueError("days_until_death column required for this target.")
    df = df[df['days_until_death'].notna()].copy()
    if len(df) == 0:
        raise ValueError("No rows with non-null days_until_death.")
    df['_target'] = (df['days_until_death'] >= 0) & (df['days_until_death'] <= near_death_days)
    df['_target'] = df['_target'].map({True: NEAR_DEATH_LABEL, False: FAR_DEATH_LABEL})
    X = df[feature_cols].copy()
    y = df['_target'].copy()
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask].copy()
    y = y[mask].copy()
    df = df.loc[mask].copy()
    X.attrs['fly_id'] = df['fly_id'].values
    X.attrs['feature_cols'] = feature_cols
    return X, y, feature_cols


def split_by_fly(X, y, test_size=0.1, val_size=0.1, random_state=42, positive_classes=None):
    """
    Split into train/val/test by fly (no fly appears in more than one set).
    Stratify by fly-level "ever positive" (proportion of flies with at least one positive-class row).
    positive_classes: list of y values considered positive (e.g. ['dying','dead'] or ['near_death']). Default ['dying','dead'].
    """
    if positive_classes is None:
        positive_classes = ['dying', 'dead']
    fly_ids = X.attrs.get('fly_id')
    if fly_ids is None:
        raise ValueError("X must have fly_id in attrs (from prepare_X_y).")
    fly_df = pd.DataFrame({'fly_id': fly_ids, 'y': y.values})
    fly_label = fly_df.groupby('fly_id')['y'].apply(
        lambda s: 1 if s.isin(positive_classes).any() else 0
    ).reset_index()
    fly_label.columns = ['fly_id', 'ever_non_alive']
    unique_flies = np.array(fly_label['fly_id'].tolist())
    strat = fly_label['ever_non_alive'].values
    # Split flies: first train+val vs test, then train vs val
    try:
        flies_train_val, flies_test = train_test_split(
            unique_flies, test_size=test_size, random_state=random_state,
            stratify=strat
        )
    except ValueError:
        flies_train_val, flies_test = train_test_split(
            unique_flies, test_size=test_size, random_state=random_state
        )
    strat_val = fly_label.set_index('fly_id').loc[flies_train_val, 'ever_non_alive'].values
    try:
        flies_train, flies_val = train_test_split(
            flies_train_val, test_size=val_size / (1 - test_size), random_state=random_state,
            stratify=strat_val
        )
    except ValueError:
        flies_train, flies_val = train_test_split(
            flies_train_val, test_size=val_size / (1 - test_size), random_state=random_state
        )
    train_mask = np.isin(fly_ids, flies_train)
    val_mask = np.isin(fly_ids, flies_val)
    test_mask = np.isin(fly_ids, flies_test)
    X_train = X[train_mask].copy()
    X_val = X[val_mask].copy()
    X_test = X[test_mask].copy()
    y_train = y[train_mask].copy()
    y_val = y[val_mask].copy()
    y_test = y[test_mask].copy()
    # Preserve attrs for fly_id on splits if needed
    for Z in (X_train, X_val, X_test):
        Z.attrs['feature_cols'] = X.attrs.get('feature_cols', [])
    print(f"  Train: {len(X_train)} rows, {len(flies_train)} flies")
    print(f"  Val:   {len(X_val)} rows, {len(flies_val)} flies")
    print(f"  Test:  {len(X_test)} rows, {len(flies_test)} flies")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgboost(X_train, y_train, X_val, y_val, num_class=None, no_tune=False, random_state=42):
    """Train XGBoost multiclass classifier. Optional tuning on val.
    Uses only classes present in y_train (e.g. alive/dying or alive/dying/dead).
    """
    # Use classes actually present in training data (sklearn requires classes ⊆ y_train)
    classes_present = np.array(sorted(y_train.unique()))
    label_map = {c: i for i, c in enumerate(classes_present)}
    n_class = len(classes_present)
    y_train_enc = y_train.map(label_map)
    y_val_enc = y_val.map(label_map)
    weights = compute_class_weight(
        'balanced', classes=classes_present, y=y_train
    )
    sample_weights = np.array([weights[list(classes_present).index(yt)] for yt in y_train])
    if no_tune:
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=n_class,
            max_depth=6,
            n_estimators=200,
            learning_rate=0.1,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='mlogloss',
        )
        model.fit(
            X_train, y_train_enc,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val_enc)],
            verbose=False,
        )
    else:
        # Simple grid over a few options
        best_score = -1
        best_model = None
        for max_depth in [4, 6, 8]:
            for lr in [0.05, 0.1]:
                m = xgb.XGBClassifier(
                    objective='multi:softmax',
                    num_class=n_class,
                    max_depth=max_depth,
                    n_estimators=200,
                    learning_rate=lr,
                    random_state=random_state,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                )
                m.fit(
                    X_train, y_train_enc,
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val_enc)],
                    verbose=False,
                )
                s = m.score(X_val, y_val_enc)
                if s > best_score:
                    best_score = s
                    best_model = m
        model = best_model
        print(f"  Best validation accuracy: {best_score:.4f}")
    return model, label_map


def evaluate_model(model, X_test, y_test, label_map, output_dir):
    """Evaluate on test set; save metrics and confusion matrix."""
    y_pred_enc = model.predict(X_test)
    inv_map = {v: k for k, v in label_map.items()}
    y_pred = pd.Series([inv_map.get(int(x), inv_map[0]) for x in y_pred_enc], index=y_test.index)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test Weighted F1: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    # Use class order from model (2-class or 3-class depending on data)
    class_order = sorted(label_map.keys(), key=lambda k: label_map[k])
    cm = confusion_matrix(y_test, y_pred, labels=class_order)
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'classification_results.txt')
    with open(results_file, 'w') as f:
        f.write("Death Prediction (XGBoost) – Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Weighted Precision: {precision:.4f}\n")
        f.write(f"Weighted Recall: {recall:.4f}\n")
        f.write(f"Weighted F1: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nConfusion matrix (rows=true, cols=pred), order: {class_order}\n")
        f.write(str(cm))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_order, yticklabels=class_order)
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved {results_file} and confusion_matrix.png")
    return {'accuracy': accuracy, 'f1': f1, 'confusion_matrix': cm}


def run_shap(model, X_train, X_test, feature_names, output_dir, max_background=200, max_test=500):
    """SHAP TreeExplainer; save summary plot (bar for multi-class)."""
    if not SHAP_AVAILABLE:
        print("  SHAP not installed; skipping SHAP plots.")
        return
    bg = X_train if len(X_train) <= max_background else X_train.sample(n=max_background, random_state=42)
    explainer = shap.TreeExplainer(model, bg)
    test = X_test if len(X_test) <= max_test else X_test.sample(n=max_test, random_state=42)
    shap_vals = explainer.shap_values(test)
    plt.figure(figsize=(10, 8))
    # For multi-class, shap_vals is a list; use bar plot for global importance
    if isinstance(shap_vals, list):
        # Mean absolute SHAP across classes for overall importance
        mean_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        mean_imp = np.mean(mean_abs, axis=0)
        order = np.argsort(mean_imp)[::-1]
        plt.barh(range(len(feature_names)), mean_imp[order])
        plt.yticks(range(len(feature_names)), [feature_names[i] for i in order])
        plt.xlabel('Mean |SHAP|')
        plt.title('SHAP feature importance (death prediction)')
    else:
        shap.summary_plot(shap_vals, test, feature_names=feature_names, show=False, plot_type='bar')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved shap_summary.png")


def main():
    parser = argparse.ArgumentParser(
        description='Death prediction: XGBoost on sliding-window features (alive/dying/dead)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--experiment-id', type=int, default=None,
                        help='Load from DB by experiment_id (default: latest)')
    parser.add_argument('--sliding-window-csv', type=str, default=None,
                        help='Load from CSV instead of DB')
    parser.add_argument('--test-size', type=float, default=0.1, help='Test proportion (default: 0.1)')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation proportion (default: 0.1)')
    parser.add_argument('--no-tune', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--exclude-longest-zero', action='store_true',
                        help='Exclude longest_zero_hours from features (reduce pre-bias)')
    parser.add_argument('--target', type=str, default='auto', choices=['auto', 'status', 'near_death'],
                        help='Target: status (alive/dying/dead), near_death (binary from days_until_death), or auto (near_death if column present else status)')
    parser.add_argument('--near-death-days', type=int, default=2,
                        help='Horizon for near_death target: 1 if 0 <= days_until_death <= N (default: 2)')
    parser.add_argument('--output-dir', type=str, default='analysis_results/death_prediction',
                        help='Output directory (default: analysis_results/death_prediction)')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    if not XGB_AVAILABLE:
        print("xgboost is required. Install with: pip install xgboost")
        sys.exit(1)

    if args.sliding_window_csv:
        df = load_sliding_window_from_csv(args.sliding_window_csv)
    else:
        df = load_sliding_window_from_db(args.experiment_id)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    use_near_death = args.target == 'near_death' or (args.target == 'auto' and 'days_until_death' in df.columns and df['days_until_death'].notna().any())
    if use_near_death:
        try:
            print(f"\n[Prepare] Building X, y from days_until_death (near_death if 0<={args.near_death_days}d)...")
            X, y, feature_cols = prepare_X_y_from_days_until_death(
                df, near_death_days=args.near_death_days, exclude_longest_zero=args.exclude_longest_zero
            )
            positive_classes = [NEAR_DEATH_LABEL]
        except ValueError as e:
            if args.target == 'near_death':
                raise
            print(f"  Falling back to status target: {e}")
            use_near_death = False
    if not use_near_death:
        print("\n[Prepare] Building X, y (alive/dying/dead)...")
        X, y, feature_cols = prepare_X_y(df, exclude_longest_zero=args.exclude_longest_zero)
        positive_classes = ['dying', 'dead']
    print(f"  Features: {feature_cols}")
    print(f"  Class counts: {y.value_counts().to_dict()}")

    print("\n[Split] Splitting by fly (train/val/test)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_by_fly(
        X, y, test_size=args.test_size, val_size=args.val_size, random_state=args.random_state,
        positive_classes=positive_classes
    )

    print("\n[Train] XGBoost multiclass...")
    model, label_map = train_xgboost(
        X_train, y_train, X_val, y_val, no_tune=args.no_tune, random_state=args.random_state
    )

    print("\n[Evaluate] Test set...")
    evaluate_model(model, X_test, y_test, label_map, out_dir)

    print("\n[SHAP] Feature importance...")
    run_shap(model, X_train, X_test, feature_cols, out_dir)

    print("\n" + "=" * 60)
    print("Death prediction analysis complete.")
    print(f"Outputs: {out_dir}/")
    print("  - classification_results.txt")
    print("  - confusion_matrix.png")
    print("  - shap_summary.png")


if __name__ == '__main__':
    main()
