# Goal | Divide a dataset into training, validation, and test sets
# Issue | Dataset is time based -> certian time periods may have extra noise so selecticing a random sample may not be ideal
# Fix | Use Walk forward validation AND purged K-Fold cross-validation


import pandas as pd
from sqlalchemy import create_engine
#from sklearn.model_selection import KFold
#from sklearn.model_selection import TimeSeriesSplit


# Two methods -> KFold and TimeSeriesSplit
# Data input is going to be .xlsx file for now


def initialize_data_pointer(FILE_LOCATION):
    data_set = pd.read_excel(FILE_LOCATION, skiprows=5, header=0)
    data_set = data_set.loc[:, ~data_set.columns.str.contains(r"^Unnamed")]
    data_set = data_set[data_set['Ticker'].notna()].reset_index(drop=True)
    return data_set

def format_to_sql(data_set, file_location):
    engine = create_engine(f"sqlite:///{file_location}")
    # write an auto-incrementing 'id' column as primary key
    data_set.to_sql('financial_data', con=engine, if_exists='replace', index=True, index_label='id')


FILE_LOCATION = "data_sets/FinancialMarketData.xlsx"
data_set = initialize_data_pointer(FILE_LOCATION)
format_to_sql(data_set, 'data_sets/all_data.db')
