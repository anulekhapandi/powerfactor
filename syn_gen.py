import pandas as pd
import numpy as np
import os

# Set your absolute path here 
output_dir = r"D:\powerfactor\data"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "synthetic_power_factor_data.csv")

# Generate datetime range
timestamps = pd.date_range(start="2025-04-01", periods=500, freq="min")

def generate_pf_series(normal_val, noise=0.01, anomaly_prob=0.05):
    series = []
    for _ in range(len(timestamps)):
        if np.random.rand() < anomaly_prob:
            # Inject anomaly (spike or dip)
            val = normal_val + np.random.choice([-0.2, -0.15, 0.2, 0.25])
        else:
            val = np.clip(np.random.normal(normal_val, noise), 0.8, 1.0)
        series.append(round(val, 3))
    return series

# Create the synthetic dataframe
df = pd.DataFrame({
    "Date": timestamps.date,
    "Time": timestamps.time,
    "Power Factor AN Avg": generate_pf_series(0.98),
    "Power Factor BN Avg": generate_pf_series(0.91),
    "Power Factor CN Avg": generate_pf_series(0.99),
    "Power Factor Total Avg": generate_pf_series(0.95)
})

# Save to CSV in specified path
df.to_csv(output_file, index=False)

print(f"Synthetic data saved at: {output_file}")
