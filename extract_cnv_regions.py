import pandas as pd
import os

# ==========================================
# CONFIG
# ==========================================

DATA_FOLDER = "D:/final internship/Raw data/"

OUTLIER_SAMPLES = [
    "GSE228150_PL1362",
    "GSE228150_PL142",
    "GSE228150_PL228",
    "GSE228150_PL250"
]

GAIN_THRESHOLD = 0.3
LOSS_THRESHOLD = -0.3

MAX_GAP = 1_000_000      # 1 Mb
MIN_MARKERS = 10

all_regions = []

# ==========================================
# PROCESS EACH SAMPLE
# ==========================================

for sample in OUTLIER_SAMPLES:

    file_path = os.path.join(
        DATA_FOLDER,
        sample + ".adj.txt"
    )

    print(f"\nProcessing {sample}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        low_memory=False
    )

    # --------------------------------------
    # Keep required columns only
    # --------------------------------------

    cols = [
        "Chr",
        "Position",
        "Affected.Log.R.Ratio"
    ]

    df = df[cols].copy()

    # --------------------------------------
    # Clean chromosome column
    # --------------------------------------

    df["Chr"] = (
        df["Chr"]
        .astype(str)
        .str.strip()
        .str.replace("chr", "", case=False, regex=False)
    )

    # Keep only autosomes 1-22

    valid_chr = [str(i) for i in range(1, 23)]

    df = df[df["Chr"].isin(valid_chr)]

    # Convert types

    df["Chr"] = df["Chr"].astype(int)

    df["Position"] = pd.to_numeric(
        df["Position"],
        errors="coerce"
    )

    df["Affected.Log.R.Ratio"] = pd.to_numeric(
        df["Affected.Log.R.Ratio"],
        errors="coerce"
    )

    df = df.dropna()

    print("Rows after cleaning:", len(df))

    # --------------------------------------
    # Detect CNVs
    # --------------------------------------

    for cnv_type, mask in [

        (
            "Loss",
            df["Affected.Log.R.Ratio"] < LOSS_THRESHOLD
        ),

        (
            "Gain",
            df["Affected.Log.R.Ratio"] > GAIN_THRESHOLD
        )

    ]:

        cnv = df[mask].copy()

        cnv = cnv.sort_values(
            ["Chr", "Position"]
        )

        if len(cnv) == 0:
            continue

        current_chr = None
        start_pos = None
        end_pos = None
        marker_count = 0
        prev_pos = None

        for _, row in cnv.iterrows():

            chrom = int(row["Chr"])
            pos = int(row["Position"])

            # Start first region

            if current_chr is None:

                current_chr = chrom
                start_pos = pos
                end_pos = pos
                prev_pos = pos
                marker_count = 1

                continue

            # Same chromosome + close

            if (
                chrom == current_chr
                and (pos - prev_pos) <= MAX_GAP
            ):

                end_pos = pos
                marker_count += 1

            else:

                if marker_count >= MIN_MARKERS:

                    all_regions.append({

                        "Sample": sample,
                        "Chromosome": current_chr,
                        "Start": start_pos,
                        "End": end_pos,
                        "Length_bp": end_pos - start_pos,
                        "Markers": marker_count,
                        "Type": cnv_type

                    })

                # Start new region

                current_chr = chrom
                start_pos = pos
                end_pos = pos
                prev_pos = pos
                marker_count = 1

            prev_pos = pos

        # Save last region

        if marker_count >= MIN_MARKERS:

            all_regions.append({

                "Sample": sample,
                "Chromosome": current_chr,
                "Start": start_pos,
                "End": end_pos,
                "Length_bp": end_pos - start_pos,
                "Markers": marker_count,
                "Type": cnv_type

            })

# ==========================================
# SAVE
# ==========================================

regions_df = pd.DataFrame(all_regions)

regions_df = regions_df.sort_values(
    ["Chromosome", "Start"]
)

regions_df.to_csv(
    "cnv_regions_outliers.csv",
    index=False
)

print("\n===================================")
print("CNV REGIONS EXTRACTED")
print("===================================")

print(regions_df.head(20))

print("\nShape:", regions_df.shape)

print("\nChromosome counts:")
print(
    regions_df["Chromosome"]
    .value_counts()
    .sort_index()
)

print("\nSaved:")
print("cnv_regions_outliers.csv")