# src/generate_saas_data.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid
import random
import os

RNG = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__),"..","data")
os.makedirs(OUT, exist_ok=True)

n_users = 20000
start_date = datetime(2024,1,1)
end_date = datetime(2024,6,30)
days = (end_date - start_date).days + 1

user_ids = [str(uuid.uuid4()) for _ in range(n_users)]
plans = ['trial','basic','pro']
signup_dates = [start_date + timedelta(days=int(RNG.integers(0, days))) for _ in range(n_users)]
user_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': signup_dates,
    'plan': RNG.choice(plans, size=n_users, p=[0.15,0.6,0.25]),
    'country': RNG.choice(['US','IN','GB','CA','DE'], size=n_users, p=[0.5,0.2,0.1,0.1,0.1])
})
user_df.to_csv(os.path.join(OUT,'users.csv'), index=False)

rows=[]
for i, uid in enumerate(user_ids):
    signup = user_df.loc[i,'signup_date']
    base = {'trial':0.3, 'basic':1.0, 'pro':1.8}[user_df.loc[i,'plan']]
    for d in range((end_date - signup).days+1):
        day = signup + timedelta(days=d)
        p_active = min(0.9, base * (0.3 + RNG.normal(0,0.05)))
        if RNG.random() < p_active:
            sessions = max(1, int(abs(RNG.normal(base*2, 1.5))))
            avg_session_min = max(1, abs(RNG.normal(10*base, 5)))
            rows.append((uid, day.date().isoformat(), sessions, avg_session_min))
event_df = pd.DataFrame(rows, columns=['user_id','event_date','sessions','avg_session_minutes'])
event_df.to_csv(os.path.join(OUT,'events.csv'), index=False)

brows=[]
for uid in user_ids:
    plan = user_df.loc[user_df['user_id']==uid,'plan'].values[0]
    for m in range(6):
        bill_date = (datetime(2024,1,1) + timedelta(days=30*m)).date()
        price = {'trial':0.0,'basic':30,'pro':80}[plan]
        brows.append((uid, bill_date.isoformat(), plan, price))
billing_df = pd.DataFrame(brows, columns=['user_id','bill_date','plan','price'])
billing_df.to_csv(os.path.join(OUT,'billing.csv'), index=False)

srows=[]
for uid in user_ids:
    if RNG.random() < 0.05:
        tcount = RNG.integers(1,6)
        for _ in range(int(tcount)):
            tdate = start_date + timedelta(days=int(RNG.integers(0, days)))
            severity = RNG.choice(['low','medium','high'], p=[0.7,0.25,0.05])
            srows.append((uid, tdate.date().isoformat(), severity))
support_df = pd.DataFrame(srows, columns=['user_id','ticket_date','severity'])
support_df.to_csv(os.path.join(OUT,'support.csv'), index=False)

# compute label (simple rule with noise)
last_day = end_date.date()
ev = event_df.copy()
ev['event_date'] = pd.to_datetime(ev['event_date'])
recent = ev[ev['event_date'] >= (end_date - timedelta(days=30))]
act_counts = recent.groupby('user_id').agg({'sessions':'sum','avg_session_minutes':'mean'}).reset_index()
act_counts.columns = ['user_id','sessions_30d','avg_session_minutes_30d']
u = user_df.merge(act_counts, on='user_id', how='left').fillna(0)
ticket_counts = support_df.groupby('user_id').size().rename('tickets').reset_index()
u = u.merge(ticket_counts, on='user_id', how='left').fillna({'tickets':0})
def churn_prob(row):
    p = 0.05
    if row['plan']=='trial': p += 0.15
    p += max(0, 0.2 - 0.02*row['sessions_30d'])
    p += 0.03*min(row['tickets'],5)
    return min(0.95, p)
u['churn_prob'] = u.apply(churn_prob, axis=1)
u['churn_label'] = u['churn_prob'].apply(lambda x: 1 if RNG.random() < x else 0)
labels = u[['user_id','churn_label','churn_prob','sessions_30d','tickets']]
labels.to_csv(os.path.join(OUT,'labels.csv'), index=False)

print("Created data in ./data: users.csv, events.csv, billing.csv, support.csv, labels.csv")
