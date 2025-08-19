import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib, os, sys
import json

DATAPATH = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURES = ["sessions_30d","avg_session_minutes_30d","tickets_30d","days_since_last_login","account_age_days"]

df_path = os.path.join(DATAPATH, "features_table.csv")
if not os.path.exists(df_path):
    sys.exit("❌ features_table.csv not found. Run: python src/features.py")

df = pd.read_csv(df_path)
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    print("❌ Missing columns in features_table.csv:", missing)
    print("🛠️  Re-run: python src/features.py (watch its printed columns).")
    sys.exit(1)

X = df[FEATURES].fillna(0)
y = df["churn_label"].fillna(0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42, test_size=0.2
)

clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)[:, 1]

print("\n===== Classification Report =====")
report = classification_report(y_test, pred, output_dict=True)
print(classification_report(y_test, pred))
auc = roc_auc_score(y_test, proba)
print("AUC:", auc)

metrics = {"auc": auc, "classification_report": report}
with open(os.path.join(DATAPATH, "performance_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

joblib.dump(clf, os.path.join(DATAPATH, "churn_rf.joblib"))
fi = pd.DataFrame({
    "feature": FEATURES,
    "importance": clf.feature_importances_
}).sort_values("importance", ascending=False)
fi.to_csv(os.path.join(DATAPATH, "feature_importances.csv"), index=False)

print("\n✅ Saved model to data/churn_rf.joblib")
print("✅ Saved importances to data/feature_importances.csv")
print("✅ Saved performance metrics to data/performance_metrics.json")