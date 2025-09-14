import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .modeling import metrics

def save_actual_vs_pred(df, y_col, yhat_col, path):
    plt.figure()
    plt.plot(df[y_col].values, label='Actual')
    plt.plot(df[yhat_col].values, label='Predicted')
    plt.legend(); plt.title('Actual vs Predicted'); plt.xlabel('Weeks'); plt.ylabel(y_col)
    plt.tight_layout(); plt.savefig(path); plt.close()

def save_residual_plots(resid, prefix):
    # time
    plt.figure()
    plt.plot(resid)
    plt.title('Residuals over time'); plt.xlabel('Weeks'); plt.ylabel('Residual')
    plt.tight_layout(); plt.savefig(f"{prefix}_time.png"); plt.close()
    # hist
    plt.figure()
    plt.hist(resid, bins=30)
    plt.title('Residual distribution'); plt.xlabel('Residual')
    plt.tight_layout(); plt.savefig(f"{prefix}_hist.png"); plt.close()

def summarize(y_true, y_pred):
    return metrics(y_true, y_pred)
