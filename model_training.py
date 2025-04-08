import pandas as pd
import numpy
import joblib
import os
from sklearn.ensemble import IsolationForest

# === File paths (customize only if needed) ===
data_path = r"D:\powerfactor\data\synthetic_power_factor_data.csv"
model_path = r"D:\powerfactor\model\isolation_forest_pf_model.pkl"

# === Load dataset ===
df = pd.read_csv(data_path)

# === Select columns for training ===
selected_columns = [
    "Power Factor AN Avg",
    "Power Factor BN Avg",
    "Power Factor CN Avg",
    "Power Factor Total Avg"
]

# === Clean and prepare data ===
X = df[selected_columns]

# === Initialize and train Isolation Forest ===
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X)

# === Save the trained model ===
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)

print(f"✅ Model trained and saved at: {model_path}")
