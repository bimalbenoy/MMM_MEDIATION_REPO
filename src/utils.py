import numpy as np
import pandas as pd
from typing import Tuple

def add_time_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.sort_values(date_col).copy()
    df['t'] = np.arange(len(df))
    return df

def add_fourier_terms(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    df = df.copy()
    t = np.arange(len(df))
    period = 52.0  # weeks
    for i in range(1, k+1):
        df[f'sin_{i}'] = np.sin(2*np.pi*i*t/period)
        df[f'cos_{i}'] = np.cos(2*np.pi*i*t/period)
    return df

def make_lags(df: pd.DataFrame, cols, max_lag: int) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        for lag in range(1, max_lag+1):
            df[f'{c}_lag{lag}'] = df[c].shift(lag)
    df.dropna(inplace=True)
    return df

def train_valid_split_by_weeks(df: pd.DataFrame, date_col: str, test_weeks: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(date_col)
    split_idx = len(df) - test_weeks
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
