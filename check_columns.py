import pandas as pd

df = pd.read_csv("autism_synthetic_scaled_final.csv")
print("🧾 Columns in dataset:")
print(df.columns.tolist())
