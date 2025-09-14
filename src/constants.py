# src/constants.py

# keep this as a *placeholder* path; we override in the notebook anyway
DATA_PATH = r"C:\Users\bimal\Downloads\mmm_mediation_repo\mmm_mediation_repo\data\weekly_data.csv"

DATE_COL = "week"
Y_COL = "revenue"

GOOGLE_COL = "google_spend"

SOCIAL_COLS = ["facebook_spend", "tiktok_spend", "instagram_spend", "snapchat_spend"]

DR_COLS = ["emails_send", "sms_send"]

CONTROL_COLS = ["average_price", "social_followers", "promotions"]

LAGS_STAGE1 = 2
LAGS_STAGE2 = 2
FOURIER_K   = 4
TEST_SPLIT_WEEKS = 26
N_FOLDS = 5
SEED = 42
