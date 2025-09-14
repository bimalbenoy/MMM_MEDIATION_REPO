import numpy as np
import pandas as pd

def adstock_geometric(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    carry = 0.0
    for i, v in enumerate(x):
        out[i] = v + alpha * carry
        carry = out[i]
    return out

def saturation_log1p(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log1p(beta * np.maximum(x, 0.0))

def apply_channel_transforms(df: pd.DataFrame, cols, alpha=0.5, beta=1.0, prefix='tr_'):
    df = df.copy()
    for c in cols:
        ad = adstock_geometric(df[c].fillna(0).values, alpha=alpha)
        sat = saturation_log1p(ad, beta=beta)
        df[f'{prefix}{c}'] = sat
    return df
