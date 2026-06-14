import os
import glob
import numpy as np
import pandas as pd

# ==========================
# CONFIG
# ==========================

DATA_FOLDER = "D:/final internship/Raw data"
OUTPUT_FILE = "master_dataset.csv"

files = os.listdir(DATA_FOLDER)

print("Number of files:", len(files))

for f in files[:20]:
    print(f)

GAIN_THRESHOLD = 0.3
LOSS_THRESHOLD = -0.3

EXTREME_GAIN = 1.0
EXTREME_LOSS = -1.0

# ==========================
# FEATURE EXTRACTION
# ==========================

def extract_features(filepath):

    sample_name = os.path.basename(filepath).replace(".adj", "")

    df = pd.read_csv(filepath, sep="\t")

    # --------------------------------
    # Core columns
    # --------------------------------

    lrr = pd.to_numeric(
        df["Affected.Log.R.Ratio"],
        errors="coerce"
    )

    baf = pd.to_numeric(
        df["Affected.B.Allele.Freq"],
        errors="coerce"
    )

    chr_col = pd.to_numeric(
        df["Chr"],
        errors="coerce"
    )

    # --------------------------------
    # Basic cleanup
    # --------------------------------

    lrr = lrr.dropna()
    baf = baf.dropna()

    total_markers = len(df)

    # --------------------------------
    # LRR Features
    # --------------------------------

    mean_lrr = lrr.mean()
    median_lrr = lrr.median()
    std_lrr = lrr.std()
    min_lrr = lrr.min()
    max_lrr = lrr.max()

    # --------------------------------
    # CNV Features
    # --------------------------------

    gain_count = (lrr > GAIN_THRESHOLD).sum()
    loss_count = (lrr < LOSS_THRESHOLD).sum()

    gain_pct = gain_count / total_markers
    loss_pct = loss_count / total_markers

    extreme_gain_count = (lrr > EXTREME_GAIN).sum()
    extreme_loss_count = (lrr < EXTREME_LOSS).sum()

    # --------------------------------
    # BAF Features
    # --------------------------------

    mean_baf = baf.mean()
    std_baf = baf.std()
    var_baf = baf.var()

    homozygous_count = (
        ((baf <= 0.1) | (baf >= 0.9))
    ).sum()

    heterozygous_count = (
        ((baf > 0.4) & (baf < 0.6))
    ).sum()

    # --------------------------------
    # BAF Anomalies
    # --------------------------------

    baf_anomaly_count = (
        ((baf > 0.1) & (baf < 0.4))
        |
        ((baf > 0.6) & (baf < 0.9))
    ).sum()

    # --------------------------------
    # High LRR deviation
    # --------------------------------

    high_deviation_count = (
        np.abs(lrr) > 0.5
    ).sum()

    combined_anomaly_count = (
        ((np.abs(lrr) > 0.5))
    ).sum()

    features = {

        "Sample": sample_name,

        # -------------------------
        # LRR
        # -------------------------
        "Mean_LRR": mean_lrr,
        "Median_LRR": median_lrr,
        "Std_LRR": std_lrr,
        "Min_LRR": min_lrr,
        "Max_LRR": max_lrr,

        # -------------------------
        # CNV
        # -------------------------
        "Gain_Count": gain_count,
        "Loss_Count": loss_count,
        "Gain_Percent": gain_pct,
        "Loss_Percent": loss_pct,
        "Extreme_Gain_Count": extreme_gain_count,
        "Extreme_Loss_Count": extreme_loss_count,

        # -------------------------
        # BAF
        # -------------------------
        "Mean_BAF": mean_baf,
        "Std_BAF": std_baf,
        "Var_BAF": var_baf,
        "Homozygous_Count": homozygous_count,
        "Heterozygous_Count": heterozygous_count,

        # -------------------------
        # Instability
        # -------------------------
        "BAF_Anomaly_Count": baf_anomaly_count,
        "High_LRR_Deviation_Count": high_deviation_count,
        "Combined_Anomaly_Count": combined_anomaly_count
    }

    # --------------------------------
    # Chromosome Features
    # --------------------------------

    for chrom in range(1, 23):

        chr_mask = chr_col == chrom

        chr_lrr = lrr[chr_mask]

        if len(chr_lrr) > 0:
            features[f"Chr{chrom}_MeanLRR"] = chr_lrr.mean()
        else:
            features[f"Chr{chrom}_MeanLRR"] = np.nan

    return features

# ==========================
# MAIN
# ==========================

all_files = sorted(
    glob.glob(os.path.join(DATA_FOLDER, "*.adj.txt"))
)

all_features = []

print(f"Found {len(all_files)} files")

for i, file in enumerate(all_files, start=1):

    print(f"[{i}/{len(all_files)}] Processing {os.path.basename(file)}")

    try:
        feat = extract_features(file)
        all_features.append(feat)

    except Exception as e:
        print("ERROR:", file)
        print(e)

master_df = pd.DataFrame(all_features)

master_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDone.")
print(master_df.shape)
print(f"Saved to {OUTPUT_FILE}")