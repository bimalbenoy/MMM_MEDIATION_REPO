import numpy as np, pandas as pd
from datetime import date, timedelta

np.random.seed(42)

weeks = 104
start = date(2023,1,2)  # Monday start
dates = [start + timedelta(weeks=i) for i in range(weeks)]

# base trends & seasonality
t = np.arange(weeks)
season = 1 + 0.1*np.sin(2*np.pi*t/52) + 0.05*np.cos(2*np.pi*2*t/52)

followers = 50000 + (t*300) + np.random.normal(0, 500, size=weeks)
avg_price = 50 + 2*np.sin(2*np.pi*t/26) + np.random.normal(0, 0.4, size=weeks)
promotion_flag = (np.random.rand(weeks) < 0.25).astype(int)

facebook = np.maximum(0, 20000 + 300*np.sin(2*np.pi*t/13) + np.random.normal(0, 1500, weeks))
tiktok   = np.maximum(0, 15000 + 400*np.cos(2*np.pi*t/17) + np.random.normal(0, 1200, weeks))
snapchat = np.maximum(0,  8000 + 200*np.sin(2*np.pi*t/11) + np.random.normal(0, 800, weeks))

# mediator: google spend partly driven by social
google_base = 18000 + 0.15*facebook + 0.12*tiktok + 0.1*snapchat + np.random.normal(0, 1500, weeks)
google = np.maximum(0, google_base)

email_sends = np.maximum(0, 200000 + 3000*np.sin(2*np.pi*t/8) + np.random.normal(0, 5000, weeks))
sms_sends   = np.maximum(0,  80000 + 2000*np.cos(2*np.pi*t/10) + np.random.normal(0, 2000, weeks))

# revenue from all, with negative sensitivity to price + promo lift
rev = ( 5.0*np.log1p(google)
      + 2.0*np.log1p(facebook) + 1.8*np.log1p(tiktok) + 1.2*np.log1p(snapchat)
      + 0.0008*email_sends + 0.0005*sms_sends
      - 40*avg_price + 0.0004*followers
      + 500*promotion_flag
) * season + np.random.normal(0, 500, weeks)

df = pd.DataFrame({
    'week': pd.to_datetime(dates),
    'revenue': rev.round(2),
    'google_spend': google.round(2),
    'facebook_spend': facebook.round(2),
    'tiktok_spend': tiktok.round(2),
    'snapchat_spend': snapchat.round(2),
    'email_sends': email_sends.astype(int),
    'sms_sends': sms_sends.astype(int),
    'avg_price': avg_price.round(2),
    'followers': followers.astype(int),
    'promotion_flag': promotion_flag
})

df.to_csv('data/weekly_data.csv', index=False)
print('Wrote data/weekly_data.csv with', len(df), 'rows')
