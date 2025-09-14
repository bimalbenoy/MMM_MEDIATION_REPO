import pandas as pd
import numpy as np
from .constants import DATE_COL, Y_COL, GOOGLE_COL, SOCIAL_COLS, DR_COLS, CONTROL_COLS, FOURIER_K, LAGS_STAGE1, LAGS_STAGE2
from .utils import add_time_index, add_fourier_terms, make_lags

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)
    # fills
    for c in SOCIAL_COLS + [GOOGLE_COL] + DR_COLS:
        if c in df:
            df[c] = df[c].fillna(0.0)
    for c in CONTROL_COLS:
        if c in df:
            df[c] = df[c].interpolate().ffill().bfill()
    return df

def stage1_design(df: pd.DataFrame) -> pd.DataFrame:
    df1 = add_time_index(df, DATE_COL)
    df1 = add_fourier_terms(df1, k=FOURIER_K)
    use_cols = SOCIAL_COLS + CONTROL_COLS
    df1 = make_lags(df1, use_cols, max_lag=LAGS_STAGE1)
    return df1

def stage2_design(df: pd.DataFrame) -> pd.DataFrame:
    df2 = add_time_index(df, DATE_COL)
    df2 = add_fourier_terms(df2, k=FOURIER_K)
    for c in CONTROL_COLS + ['google_pred', 'google_resid']:
        if c in df2:
            for lag in range(1, LAGS_STAGE2+1):
                df2[f'{c}_lag{lag}'] = df2[c].shift(lag)
    df2.dropna(inplace=True)
    return df2
