import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("clustered_dataset.csv")

print("Dataset Shape:", df.shape)

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop(columns=["Sample", "Cluster"])

y = df["Cluster"]

print("\nFeatures:", X.shape[1])

# ==========================================
# RANDOM FOREST
# ==========================================

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X, y)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# ==========================================
# SAVE RESULTS
# ==========================================

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nTop 15 Important Features\n")
print(importance_df.head(15))

# ==========================================
# PLOT TOP 15
# ==========================================

top15 = importance_df.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top15["Feature"][::-1],
    top15["Importance"][::-1]
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")

plt.title(
    "Top 15 Features Driving Genomic Instability Clusters"
)

plt.tight_layout()

plt.savefig(
    "feature_importance_top15.png",
    dpi=300
)

plt.show()

print("\nSaved:")
print("feature_importance.csv")
print("feature_importance_top15.png")