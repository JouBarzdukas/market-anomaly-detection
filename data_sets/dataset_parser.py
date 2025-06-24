# Goal | Divide a dataset into training, validation, and test sets
# Issue | Dataset is time based -> certain time periods may have extra noise so selecting a random sample may not be ideal
# Fix | Use Walk forward validation AND purged K-Fold cross-validation

import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import TimeSeriesSplit, KFold

PURGE = 5
# Two methods -> KFold and TimeSeriesSplit
# Data input is going to be .xlsx file for now

def initialize_data_pointer(FILE_LOCATION):
    data_set = pd.read_excel(FILE_LOCATION, sheet_name='EWS')
    data_set = data_set.loc[:, ~data_set.columns.str.contains(r"^Unnamed")]
    data_set = data_set.rename(columns={'Data': 'Date'})
    data_set['Date'] = pd.to_datetime(data_set['Date'])
    data_set = data_set.sort_values('Date').set_index('Date', drop=True)
    data_set = data_set.drop(columns=['Data'], errors='ignore')
    data_set = data_set[~data_set.index.duplicated(keep='first')]
    return data_set

def format_to_sql(data_set, file_location, table_name):
    engine = create_engine(f"sqlite:///{file_location}")
    data_set.to_sql(table_name, con=engine, if_exists='replace', index=True, index_label='Date')
    engine.dispose()

def apply_purge(training_index, validation_index, k=PURGE):
    return training_index[training_index < validation_index[0] - k]

def make_training_data(data_set, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=False)
    walk_forward = TimeSeriesSplit(n_splits=n_splits)
    return {
        'KFold':      [data_set.iloc[apply_purge(train, val)] for train, val in kfold.split(data_set)],
        'WalkForward':[data_set.iloc[apply_purge(train, val)] for train, val in walk_forward.split(data_set)]
    }

def make_validation_data(data_set, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=False)
    walk_forward = TimeSeriesSplit(n_splits=n_splits)
    return {
        'KFold':      [data_set.iloc[val] for _, val in kfold.split(data_set)],
        'WalkForward':[data_set.iloc[val] for _, val in walk_forward.split(data_set)]
    }

def push_to_sql(data_sets, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        os.remove(path)
    engine = create_engine(f"sqlite:///{path}")
    if isinstance(data_sets, list):
        for i, df in enumerate(data_sets, 1):
            df.to_sql(f"fold_{i}", con=engine, if_exists="replace", index=True, index_label="Date")
    else:
        data_sets.to_sql("financial_data", con=engine, if_exists="replace", index=True, index_label="Date")
    engine.dispose()


file_location    = "data_sets/FinancialMarketData.xlsx"
data_set         = initialize_data_pointer(file_location)
format_to_sql(data_set, 'data_sets/all_data.db', 'financial_data')

training_data    = make_training_data(data_set)
validation_data  = make_validation_data(data_set)

for train_type, data in training_data.items():
    push_to_sql(data, f"data_sets/{train_type}/{train_type}_train.db")

for valid_type, data in validation_data.items():
    push_to_sql(data, f"data_sets/{valid_type}/{valid_type}_val.db")
