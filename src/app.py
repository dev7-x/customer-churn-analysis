# src/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)
DATAPATH = os.path.join(os.path.dirname(__file__),"..","data")
MODEL_PATH = os.path.join(DATAPATH,'churn_rf.joblib')
clf = joblib.load(MODEL_PATH)

FEATURES = ['sessions_30d','avg_session_minutes_30d','tickets_30d','days_since_last_login','account_age_days']

@app.route('/score', methods=['POST'])
def score():
    payload = request.get_json()
    df = pd.DataFrame(payload if isinstance(payload, list) else [payload])
    X = df[FEATURES]
    probs = clf.predict_proba(X)[:,1]
    df['churn_prob'] = probs
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
