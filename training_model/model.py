# Goal | Build and evaluate crash-prediction models
# Issue | Need quant-ready metrics and no look-ahead
# Fix | Feature factory, purged CV, logistic & LightGBM, enhanced debugging

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, inspect
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
import joblib
import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)


PURGE = 5
ROLL_WINDOWS = [5, 10, 20]
EARLY_ALERT = 10
MODEL_DIR = Path("training_model") / "models"


def read_folds(database_path):
    engine = create_engine(f"sqlite:///{database_path}")
    table_names = inspect(engine).get_table_names()
    print(f"[DEBUG] read_folds: Found tables {table_names} in {database_path}")
    fold_frames = []
    for table_name in table_names:
        frame = (
            pd.read_sql_table(table_name, engine, parse_dates=["Date"])
            .set_index("Date", drop=True)
            .sort_index()
        )
        print(f"[DEBUG]  - loaded fold '{table_name}' with shape {frame.shape}")
        fold_frames.append(frame)
    engine.dispose()
    fold_frames.sort(key=lambda frame: frame.index.min())
    return fold_frames


def feature_factory(frame):
    print(f"[DEBUG] feature_factory: input frame shape {frame.shape}")
    numeric_columns = frame.columns.drop("Y")
    # log transform; replace zeros to avoid -inf
    log_prices = np.log(frame[numeric_columns].replace(0, np.nan))
    returns = log_prices.diff()

    vol5 = returns.rolling(ROLL_WINDOWS[0], min_periods=1).std().add_suffix("_vol5")
    vol10 = returns.rolling(ROLL_WINDOWS[1], min_periods=1).std().add_suffix("_vol10")
    vol20 = returns.rolling(ROLL_WINDOWS[2], min_periods=1).std().add_suffix("_vol20")

    raw_features = pd.concat([returns.add_suffix("_ret"), vol5, vol10, vol20], axis=1).shift(1)
    print(f"[DEBUG]  raw_features preview:\n{raw_features.head()}\n...")

    # replace inf and drop missing
    clean_features = raw_features.replace([np.inf, -np.inf], np.nan)
    total_nans = clean_features.isna().sum().sum()
    row_nans = clean_features.isna().any(axis=1).sum()
    print(f"[DEBUG]  raw_features shape {raw_features.shape}, total NaNs={total_nans}, rows with any NaN={row_nans}")

    if clean_features.empty:
        na_counts = raw_features.isna().sum(axis=1)
        print(f"[DEBUG]  raw_features NA counts per row:\n{na_counts.value_counts().head()}" )
        print(f"[DEBUG]  raw_features sample with NAs:\n{raw_features[raw_features.isna().any(axis=1)].head()}" )

    clean_features = clean_features.dropna()
    print(f"[DEBUG]  clean_features shape after dropna {clean_features.shape}")

    target = frame["Y"].loc[clean_features.index]
    print(f"[DEBUG]  target shape {target.shape}")
    return clean_features, target


def score_fold(y_true, y_prob, window_days=EARLY_ALERT):
    roc = roc_auc_score(y_true, y_prob)
    prc = average_precision_score(y_true, y_prob)
    probability_series = pd.Series(y_prob, index=y_true.index)
    lead_flag = y_true.shift(-window_days).fillna(0) > 0
    early_hits = int(((probability_series > 0.5) & lead_flag).sum())
    return roc, prc, early_hits


def run_experiment(split_name):
    print(f"\n[DEBUG] run_experiment: Starting '{split_name}'")
    train_db_path = f"data_sets/{split_name}/{split_name}_train.db"
    val_db_path = f"data_sets/{split_name}/{split_name}_val.db"

    training_folds = read_folds(train_db_path)
    validation_folds = read_folds(val_db_path)
    print(f"[DEBUG]  {len(training_folds)} training folds, {len(validation_folds)} validation folds")

    logit_pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    lightgbm_model = LGBMClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        verbose=-1
    )

    evaluation_results = {"logit": [], "lgbm": []}
    output_directory = MODEL_DIR / split_name
    output_directory.mkdir(parents=True, exist_ok=True)

    for fold_index, (train_frame, val_frame) in enumerate(zip(training_folds, validation_folds), start=1):
        print(f"\n[DEBUG] Fold {fold_index}: raw train {train_frame.shape}, raw val {val_frame.shape}")
        X_train, y_train = feature_factory(train_frame)
        X_val, y_val = feature_factory(val_frame)

        if X_train.empty or X_val.empty:
            print(f"[DEBUG]  Skipping fold {fold_index}: after preprocessing X_train {X_train.shape}, X_val {X_val.shape}")
            continue

        try:
            logit_pipeline.fit(X_train, y_train)
        except Exception as e:
            print(f"[ERROR] logit.fit failed on fold {fold_index}: {e}")
            print("[DEBUG] X_train.describe():\n", X_train.describe())
            raise

        lightgbm_model.fit(X_train, y_train)

        joblib.dump(logit_pipeline, output_directory / f"logit_fold{fold_index}.pkl")
        joblib.dump(lightgbm_model, output_directory / f"lgbm_fold{fold_index}.pkl")

        for model_name, model in [("logit", logit_pipeline), ("lgbm", lightgbm_model)]:
            y_prob = model.predict_proba(X_val)[:, 1]
            metrics = score_fold(y_val, y_prob)
            evaluation_results[model_name].append(metrics)
            print(f"[DEBUG]  {model_name} fold{fold_index} metrics: ROC={metrics[0]:.3f}, PR={metrics[1]:.3f}, early={metrics[2]}")

    for model_name, metrics in evaluation_results.items():
        if metrics:
            rocs, prcs, earlys = zip(*metrics)
        else:
            rocs, prcs, earlys = [], [], []
        print(
            f"{split_name} | {model_name} | "
            f"ROC_AUC {np.mean(rocs):.3f} ± {np.std(rocs):.3f} | "
            f"PR_AUC {np.mean(prcs):.3f} ± {np.std(prcs):.3f} | "
            f"early_hits {np.sum(earlys)}"
        )


if __name__ == "__main__":
    run_experiment("KFold")
    run_experiment("WalkForward")
