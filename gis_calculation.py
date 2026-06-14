import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("master_dataset.csv")

# Features contributing to instability
gis_features = [
    "Std_LRR",
    "Gain_Percent",
    "Loss_Percent",
    "Var_BAF",
    "High_LRR_Deviation_Count"
]

# Normalize to 0-1
scaler = MinMaxScaler()

scaled = scaler.fit_transform(df[gis_features])

scaled_df = pd.DataFrame(
    scaled,
    columns=gis_features
)

# Weighted GIS
df["GIS"] = (
    0.30 * scaled_df["Std_LRR"] +
    0.20 * scaled_df["Gain_Percent"] +
    0.20 * scaled_df["Loss_Percent"] +
    0.15 * scaled_df["Var_BAF"] +
    0.15 * scaled_df["High_LRR_Deviation_Count"]
)

# Convert to 0-100 scale
df["GIS"] = df["GIS"] * 100

# Save
df.to_csv("master_dataset_with_GIS.csv", index=False)

print(df[["Sample", "GIS"]].head())

print("\nGIS Statistics")
print(df["GIS"].describe())

df = pd.read_csv("master_dataset_with_GIS.csv")

plt.figure(figsize=(8,5))
plt.hist(df["GIS"], bins=15)

plt.xlabel("GIS")
plt.ylabel("Number of Samples")
plt.title("Genomic Instability Score Distribution")

plt.show()