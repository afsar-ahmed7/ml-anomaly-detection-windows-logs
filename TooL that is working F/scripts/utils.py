import pandas as pd
import numpy as np
from datetime import datetime

def load_data(path):
    """
    Load a CSV file and return a DataFrame.
    """
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return pd.DataFrame()

def save_data(df, path):
    """
    Save a DataFrame to CSV.
    """
    try:
        df.to_csv(path, index=False)
        print(f"✅ Data saved to: {path}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

def parse_timestamp_column(df, col_name):
    """
    Parse datetime from a specified column and return modified DataFrame.
    """
    df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
    return df

def summarize_nulls(df):
    """
    Print missing value summary.
    """
    nulls = df.isnull().sum()
    print("\n🔍 Null Values Summary:")
    print(nulls[nulls > 0])

def print_head(df, n=5, name="Data Preview"):
    """
    Print first n rows of a DataFrame.
    """
    print(f"\n📄 {name} (top {n} rows):")
    print(df.head(n))
