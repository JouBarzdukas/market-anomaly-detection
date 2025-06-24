# Goal | Validate the KFold and WalkForward SQLite datasets
# Issue | Need to confirm purge gap, no overlap, and matching fold pairs
# Fix | Read both *_train.db and *_val.db, run gap / overlap checks

import os
import pandas as pd
from sqlalchemy import create_engine, inspect

PURGE = 5


def load_folds(database_path):
    engine = create_engine(f"sqlite:///{database_path}")
    table_names = inspect(engine).get_table_names()
    folds = {}
    for table in table_names:
        frame = pd.read_sql_table(table, engine, parse_dates=["Date"])
        frame = frame.set_index("Date", drop=True).sort_index()
        folds[table] = frame
    engine.dispose()
    return folds


def check_fold_pair(training_frame, validation_frame, purge):
    if not training_frame.index.is_monotonic_increasing:
        raise ValueError("training index not ordered")
    if not validation_frame.index.is_monotonic_increasing:
        raise ValueError("validation index not ordered")
    if training_frame.index.has_duplicates:
        raise ValueError("training index contains duplicates")
    if validation_frame.index.has_duplicates:
        raise ValueError("validation index contains duplicates")
    if not training_frame.index.intersection(validation_frame.index).empty:
        raise ValueError("overlap detected between training and validation")
    gap = (validation_frame.index.min() - training_frame.index.max()).days
    if gap < purge:
        raise ValueError(f"purge gap {gap} < required {purge}")


def validate_split(split_name, purge=PURGE):
    training_db = f"data_sets/{split_name}/{split_name}_train.db"
    validation_db = f"data_sets/{split_name}/{split_name}_val.db"
    if not os.path.isfile(training_db):
        raise FileNotFoundError(f"{training_db} not found")
    if not os.path.isfile(validation_db):
        raise FileNotFoundError(f"{validation_db} not found")
    training_folds = load_folds(training_db)
    validation_folds = load_folds(validation_db)
    if set(training_folds.keys()) != set(validation_folds.keys()):
        raise ValueError(f"{split_name}: mismatched fold names")
    for fold_name in training_folds:
        check_fold_pair(training_folds[fold_name], validation_folds[fold_name], purge)
    print(f"{split_name}: {len(training_folds)} folds validated (gap ≥ {purge}, no overlap)")

validate_split("KFold")
validate_split("WalkForward")
