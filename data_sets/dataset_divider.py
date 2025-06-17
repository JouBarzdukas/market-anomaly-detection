# Goal | Divide a dataset into training, validation, and test sets
# Issue | Dataset is time based -> certian time periods may have extra noise so selecticing a random sample may not be ideal
# Fix | Use Walk forward validation AND purged K-Fold cross-validation


import pandas as pd
#from sklearn.model_selection import KFold
#from sklearn.model_selection import TimeSeriesSplit


# Two methods -> KFold and TimeSeriesSplit
# Data input is going to be .xlsx file for now


FILE_LOCATION = "data_sets/FinancialMarketData.xlsx"

def test_open_file():
    try:
        df = pd.read_excel(FILE_LOCATION)
        print("File opened successfully.")
        return df
    except Exception as e:
        print(f"Error opening file: {e}")
        return None
    

def return_single_row():
    import pandas as pd

    df = pd.read_excel(FILE_LOCATION,
                    skiprows=5,
                    header=0)
    df = df[df['Ticker'].notna()].reset_index(drop=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map(' '.join).str.strip()
    print(df.iloc[0])


    
test_open_file()
return_single_row()
