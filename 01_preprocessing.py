
# ============================================================
# PROJECT: AGRICULTURAL PRODUCTIVITY PREDICTION
# STAGE 1: DATASET VALIDATION AND QUALITY AUDIT
# ============================================================

import pandas as pd
from pathlib import Path
import gc

# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(
    r"C:\Users\NajHi\OneDrive\Desktop\lpu\sem 5\INT234 pred analysis\CA2 proj"
)

RAW_DIR = BASE_DIR / "raw_dataset"
OUTPUT_DIR = BASE_DIR / "processed_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "Groundwater": RAW_DIR / "groundwater.csv",
    "Rainfall": RAW_DIR / "rainfall_daily.csv",
    "APY": RAW_DIR / "apy.csv"
}

# ------------------------------------------------------------
# 2. INITIALISE AUDIT REPORTS
# ------------------------------------------------------------

dataset_summary = []
missing_summary = []
duplicate_summary = []

# ------------------------------------------------------------
# 3. LOAD AND INSPECT EACH DATASET
# ------------------------------------------------------------

for dataset_name, file_path in FILES.items():

    print("\n" + "=" * 65)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 65)

    df = pd.read_csv(file_path, low_memory=False)

    # Basic information
    rows, columns = df.shape

    print("Rows:", rows)
    print("Columns:", columns)
    print("\nColumn names:")
    print(df.columns)

    print("\nData types:")
    print(df.dtypes)

    # --------------------------------------------------------
    # 4. MISSING VALUE ANALYSIS
    # --------------------------------------------------------

    missing_count = df.isnull().sum()

    for column in df.columns:
        missing_summary.append({
            "Dataset": dataset_name,
            "Column": column,
            "Missing_Count": int(missing_count[column]),
            
        })

    print("\nMissing values:")
    print(
        pd.DataFrame({
            "Missing_Count": missing_count,
        })
    )

    # --------------------------------------------------------
    # 5. DUPLICATE ANALYSIS
    # --------------------------------------------------------

    exact_duplicates = int(df.duplicated().sum())

    duplicate_ids = (
        int(df["id"].duplicated().sum())
        if "id" in df.columns else None
    )

    duplicate_summary.append({
        "Dataset": dataset_name,
        "Exact_Duplicate_Rows": exact_duplicates,
        "Duplicate_ID_Rows": duplicate_ids
    })

    print("\nDuplicate analysis:")
    print("Exact duplicate rows:", exact_duplicates)

    if duplicate_ids is not None:
        print("Duplicate ID rows:", duplicate_ids)

    # --------------------------------------------------------
    # 6. DATASET SUMMARY
    # --------------------------------------------------------

    dataset_summary.append({
        "Dataset": dataset_name,
        "Total_Rows": rows,
        "Total_Columns": columns,
        "Total_Missing_Values": int(missing_count.sum()),
        "Exact_Duplicate_Rows": exact_duplicates,
        "Duplicate_ID_Rows": duplicate_ids
    })

    # Release memory before loading the next dataset
    del df
    gc.collect()


# ------------------------------------------------------------
# 7. SAVE AUDIT RESULTS
# ------------------------------------------------------------

summary_df = pd.DataFrame(dataset_summary)
missing_df = pd.DataFrame(missing_summary)
duplicates_df = pd.DataFrame(duplicate_summary)

summary_df.to_csv(
    OUTPUT_DIR / "dataset_quality_summary.csv",
    index=False
)

missing_df.to_csv(
    OUTPUT_DIR / "missing_values_audit.csv",
    index=False
)

duplicates_df.to_csv(
    OUTPUT_DIR / "duplicate_audit.csv",
    index=False
)

# ------------------------------------------------------------
# 8. DISPLAY FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL DATASET QUALITY SUMMARY")
print("=" * 65)

print(summary_df.to_string(index=False))

print("\nAudit files saved successfully in:", OUTPUT_DIR)

print("\n1. dataset_quality_summary.csv")
print("2. missing_values_audit.csv")
print("3. duplicate_audit.csv")



# ============================================================
# STAGE 2: INVESTIGATION OF MISSING VALUE PATTERNS
# ============================================================

print("\nINVESTIGATING MISSING VALUE PATTERNS")

# ------------------------------------------------------------
# 1. RAINFALL MISSING VALUES BY YEAR
# ------------------------------------------------------------

rainfall = pd.read_csv(
    FILES["Rainfall"],
    usecols=[
        "date", "state_name", "district_name",
        "actual", "rfs", "normal", "deviation"
    ],
    low_memory=False
)

rainfall["date"] = pd.to_datetime(
    rainfall["date"], errors="coerce"
)

rainfall["year"] = rainfall["date"].dt.year

rainfall_columns = [
    "actual", "rfs", "normal", "deviation"
]

rainfall_yearly = (
    rainfall.groupby("year")[rainfall_columns]
    .agg(lambda x: x.isna().sum())
)

print("\nRAINFALL MISSING VALUES BY YEAR")
print(rainfall_yearly.to_string())

rainfall_yearly.to_csv(
    OUTPUT_DIR / "rainfall_missing_by_year.csv"
)

# ------------------------------------------------------------
# 2. RAINFALL ROWS WITH ALL FOUR MEASUREMENTS MISSING
# ------------------------------------------------------------

all_rainfall_missing = rainfall[
    rainfall[rainfall_columns].isna().all(axis=1)
]

print("\nRAINFALL RECORDS WITH ALL FOUR MEASUREMENTS MISSING:")
print(len(all_rainfall_missing))

# ------------------------------------------------------------
# 3. DISTRICTS WITH MOST MISSING ACTUAL RAINFALL VALUES
# ------------------------------------------------------------

district_missing = (
    rainfall.groupby(
        ["state_name", "district_name"]
    )["actual"]
    .agg(
        Total_Records="size",
        Missing_Actual=lambda x: x.isna().sum()
    )
    .reset_index()
)

district_missing["Missing_Percentage"] = (
    district_missing["Missing_Actual"] /
    district_missing["Total_Records"] * 100
).round(2)

district_missing = district_missing.sort_values(
    "Missing_Percentage", ascending=False
)

print("\nTOP 15 DISTRICTS WITH MISSING ACTUAL RAINFALL:")
print(district_missing.head(15).to_string(index=False))

district_missing.to_csv(
    OUTPUT_DIR / "rainfall_missing_by_district.csv",
    index=False
)

del rainfall, all_rainfall_missing, district_missing
gc.collect()

# ------------------------------------------------------------
# 4. APY MISSING PRODUCTION INVESTIGATION
# ------------------------------------------------------------

apy = pd.read_csv(
    FILES["APY"],
    usecols=[
        "year", "state_name", "district_name",
        "crop_name", "season", "production", "yield"
    ],
    low_memory=False
)

print("\nAPY MISSING PRODUCTION BY YEAR")

apy_yearly = (
    apy.groupby("year")["production"]
    .agg(
        Total_Records="size",
        Missing_Production=lambda x: x.isna().sum()
    )
    .reset_index()
)

apy_yearly["Missing_Percentage"] = (
    apy_yearly["Missing_Production"] /
    apy_yearly["Total_Records"] * 100
).round(2)

print(apy_yearly.to_string(index=False))

apy_yearly.to_csv(
    OUTPUT_DIR / "apy_missing_by_year.csv",
    index=False
)

# ------------------------------------------------------------
# 5. APY MISSING PRODUCTION BY CROP AND SEASON
# ------------------------------------------------------------

apy_crop_missing = (
    apy.groupby(["crop_name", "season"])["production"]
    .agg(
        Total_Records="size",
        Missing_Production=lambda x: x.isna().sum()
    )
    .reset_index()
)

apy_crop_missing["Missing_Percentage"] = (
    apy_crop_missing["Missing_Production"] /
    apy_crop_missing["Total_Records"] * 100
).round(2)

apy_crop_missing = apy_crop_missing.sort_values(
    "Missing_Production", ascending=False
)

print("\nTOP 20 CROP-SEASON GROUPS WITH MISSING PRODUCTION:")
print(apy_crop_missing.head(20).to_string(index=False))

apy_crop_missing.to_csv(
    OUTPUT_DIR / "apy_missing_by_crop_season.csv",
    index=False
)

print("\nMISSING VALUE INVESTIGATION COMPLETED.")
print("Results saved in:", OUTPUT_DIR)

del apy, apy_yearly, apy_crop_missing
gc.collect()



# ============================================================
# STAGE 2: FIVE-YEAR DATA COVERAGE ANALYSIS
# ============================================================



print("\n" + "=" * 65)
print("FIVE-YEAR DATA COVERAGE ANALYSIS")
print("=" * 65)

# Proposed five-year period for initial comparison
START_YEAR = 2017
END_YEAR = 2021

# ------------------------------------------------------------
# 1. GROUNDWATER COVERAGE
# ------------------------------------------------------------

groundwater = pd.read_csv(
    FILES["Groundwater"],
    usecols=["date", "currentlevel", "level_diff"]
)

groundwater["date"] = pd.to_datetime(
    groundwater["date"], errors="coerce"
)

groundwater["year"] = groundwater["date"].dt.year

print("\nGROUNDWATER COVERAGE")
print("Earliest date:", groundwater["date"].min())
print("Latest date:", groundwater["date"].max())

gw_yearly = groundwater.groupby("year").agg(
    Total_Records=("date", "size"),
    Missing_Current_Level=("currentlevel", lambda x: x.isna().sum()),
    Missing_Level_Diff=("level_diff", lambda x: x.isna().sum())
)

print("\nGroundwater records by year:")
print(gw_yearly.to_string())

gw_selected = groundwater[
    groundwater["year"].between(START_YEAR, END_YEAR)
]

print("\nGroundwater selected period:")
print("Records:", len(gw_selected))
print("Date range:", gw_selected["date"].min(),
      "to", gw_selected["date"].max())

del groundwater, gw_selected
gc.collect()

# ------------------------------------------------------------
# 2. RAINFALL COVERAGE
# ------------------------------------------------------------

rainfall = pd.read_csv(
    FILES["Rainfall"],
    usecols=["date", "actual", "rfs", "normal", "deviation"]
)

rainfall["date"] = pd.to_datetime(
    rainfall["date"], errors="coerce"
)

rainfall["year"] = rainfall["date"].dt.year

print("\nRAINFALL COVERAGE")
print("Earliest date:", rainfall["date"].min())
print("Latest date:", rainfall["date"].max())

rainfall_columns = [
    "actual", "rfs", "normal", "deviation"
]

rain_yearly = rainfall.groupby("year").agg(
    Total_Records=("date", "size"),
    Actual_Missing=("actual", lambda x: x.isna().sum()),
    RFS_Missing=("rfs", lambda x: x.isna().sum()),
    Normal_Missing=("normal", lambda x: x.isna().sum()),
    Deviation_Missing=("deviation", lambda x: x.isna().sum())
)

print("\nRainfall records and missing values by year:")
print(rain_yearly.to_string())

rain_selected = rainfall[
    rainfall["year"].between(START_YEAR, END_YEAR)
]

print("\nRainfall selected period:")
print("Records:", len(rain_selected))
print("Date range:", rain_selected["date"].min(),
      "to", rain_selected["date"].max())

print("\nMissing values in selected rainfall period:")

for column in rainfall_columns:
    missing = rain_selected[column].isna().sum()
    total = len(rain_selected)
    percentage = missing / total * 100 if total else 0

    print(
        f"{column}: {missing} missing "
        f"({percentage:.2f}%)"
    )

del rainfall, rain_selected
gc.collect()

# ------------------------------------------------------------
# 3. APY COVERAGE
# ------------------------------------------------------------

apy = pd.read_csv(
    FILES["APY"],
    usecols=[
        "year", "production", "yield",
        "crop_name", "season"
    ],
    low_memory=False
)

# Extract starting year from labels such as 2018-2019
apy["start_year"] = pd.to_numeric(
    apy["year"].astype("string").str.extract(
        r"(\d{4})"
    )[0],
    errors="coerce"
)

print("\nAPY COVERAGE")
print("Earliest agricultural year:", apy["year"].min())
print("Latest agricultural year:", apy["year"].max())

apy_yearly = apy.groupby("start_year").agg(
    Total_Records=("year", "size"),
    Missing_Production=("production", lambda x: x.isna().sum()),
    Missing_Yield=("yield", lambda x: x.isna().sum())
)

print("\nAPY records by agricultural starting year:")
print(apy_yearly.to_string())

apy_selected = apy[
    apy["start_year"].between(START_YEAR, END_YEAR)
]

print("\nAPY selected agricultural years:")
print("Records:", len(apy_selected))
print("Year labels:")
print(sorted(apy_selected["year"].dropna().unique()))

print("\nMissing values in selected APY period:")

for column in ["production", "yield"]:
    missing = apy_selected[column].isna().sum()
    total = len(apy_selected)
    percentage = missing / total * 100 if total else 0

    print(
        f"{column}: {missing} missing "
        f"({percentage:.2f}%)"
    )

# ------------------------------------------------------------
# 4. SAVE COVERAGE REPORTS
# ------------------------------------------------------------

gw_yearly.to_csv(
    OUTPUT_DIR / "groundwater_yearly_coverage.csv"
)

rain_yearly.to_csv(
    OUTPUT_DIR / "rainfall_yearly_coverage.csv"
)

apy_yearly.to_csv(
    OUTPUT_DIR / "apy_yearly_coverage.csv"
)

print("\nCoverage reports saved successfully.")

del apy, apy_selected
gc.collect()

# ============================================================
# GROUNDWATER COVERAGE INVESTIGATION
# ============================================================

groundwater = pd.read_csv(
    FILES["Groundwater"],
    usecols=[
        "date",
        "state_code",
        "district_code",
        "station_name",
        "currentlevel"
    ],
    low_memory=False
)

groundwater["date"] = pd.to_datetime(
    groundwater["date"],
    errors="coerce"
)

groundwater["year"] = groundwater["date"].dt.year

groundwater = groundwater[
    groundwater["year"].between(2017, 2021)
].copy()

print("\nGROUNDWATER COVERAGE INVESTIGATION")

# 1. Records per year
print("\nRecords per year:")
print(groundwater.groupby("year").size())

# 2. Unique monitoring stations per year
print("\nUnique stations per year:")
print(
    groundwater.groupby("year")["station_name"].nunique()
)

# 3. Unique districts per year
print("\nUnique districts per year:")
print(
    groundwater.groupby("year")["district_code"].nunique()
)

# 4. Unique state-district combinations per year
print("\nUnique state-district combinations per year:")
print(
    groundwater.groupby("year")[
        ["state_code", "district_code"]
    ].apply(lambda x: x.drop_duplicates().shape[0])
)

# 5. Save investigation results
coverage = groundwater.groupby("year").agg(
    Total_Records=("date", "count"),
    Unique_Stations=("station_name", "nunique"),
    Unique_Districts=("district_code", "nunique"),
    Unique_States=("state_code", "nunique")
)

coverage.to_csv(
    OUTPUT_DIR / "groundwater_coverage_investigation.csv"
)

print("\nInvestigation report saved successfully.")

# ============================================================
# DISTRICT OVERLAP ANALYSIS: 2017–2021
# ============================================================

print("\n" + "=" * 65)
print("DISTRICT OVERLAP ANALYSIS: 2017–2021")
print("=" * 65)

# Load only the required columns
gw = pd.read_csv(
    FILES["Groundwater"],
    usecols=["date", "state_code", "district_code"]
)

rain = pd.read_csv(
    FILES["Rainfall"],
    usecols=["date", "state_code", "district_code"]
)

apy = pd.read_csv(
    FILES["APY"],
    usecols=["year", "state_code", "district_code"]
)

# Extract calendar year
gw["date"] = pd.to_datetime(gw["date"], errors="coerce")
rain["date"] = pd.to_datetime(rain["date"], errors="coerce")

gw["year"] = gw["date"].dt.year
rain["year"] = rain["date"].dt.year

apy["year"] = pd.to_numeric(
    apy["year"].astype(str).str.extract(r"(\d{4})")[0],
    errors="coerce"
)

# Filter selected period
gw = gw[gw["year"].between(2017, 2021)]
rain = rain[rain["year"].between(2017, 2021)]
apy = apy[apy["year"].between(2017, 2021)]

# Create unique state-district combinations
gw_pairs = set(
    map(tuple, gw[["state_code", "district_code"]]
        .dropna().drop_duplicates().values)
)

rain_pairs = set(
    map(tuple, rain[["state_code", "district_code"]]
        .dropna().drop_duplicates().values)
)

apy_pairs = set(
    map(tuple, apy[["state_code", "district_code"]]
        .dropna().drop_duplicates().values)
)

# Compare coverage
common_all = gw_pairs & rain_pairs & apy_pairs

print("\nUnique state-district combinations:")
print("Groundwater:", len(gw_pairs))
print("Rainfall:", len(rain_pairs))
print("APY:", len(apy_pairs))

print("\nCommon across all three datasets:", len(common_all))

print("\nGroundwater and Rainfall:", len(gw_pairs & rain_pairs))
print("Rainfall and APY:", len(rain_pairs & apy_pairs))
print("Groundwater and APY:", len(gw_pairs & apy_pairs))

print("\nDistrict overlap analysis completed.")

# ============================================================
# YEAR-WISE DISTRICT OVERLAP ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("YEAR-WISE DISTRICT OVERLAP: 2017–2021")
print("=" * 65)

# Reload only required columns
gw = pd.read_csv(
    FILES["Groundwater"],
    usecols=["date", "state_code", "district_code"]
)

rain = pd.read_csv(
    FILES["Rainfall"],
    usecols=["date", "state_code", "district_code"]
)

apy = pd.read_csv(
    FILES["APY"],
    usecols=["year", "state_code", "district_code"]
)

# Convert dates
gw["date"] = pd.to_datetime(gw["date"], errors="coerce")
rain["date"] = pd.to_datetime(rain["date"], errors="coerce")

gw["year"] = gw["date"].dt.year
rain["year"] = rain["date"].dt.year

# Extract agricultural starting year from APY
apy["year"] = pd.to_numeric(
    apy["year"].astype(str).str.extract(r"(\d{4})")[0],
    errors="coerce"
)

# Keep selected period
gw = gw[gw["year"].between(2017, 2021)]
rain = rain[rain["year"].between(2017, 2021)]
apy = apy[apy["year"].between(2017, 2021)]

results = []

for year in range(2017, 2022):

    gw_year = gw[gw["year"] == year]
    rain_year = rain[rain["year"] == year]
    apy_year = apy[apy["year"] == year]

    gw_set = set(
        map(tuple, gw_year[
            ["state_code", "district_code"]
        ].dropna().drop_duplicates().values)
    )

    rain_set = set(
        map(tuple, rain_year[
            ["state_code", "district_code"]
        ].dropna().drop_duplicates().values)
    )

    apy_set = set(
        map(tuple, apy_year[
            ["state_code", "district_code"]
        ].dropna().drop_duplicates().values)
    )

    common = gw_set & rain_set & apy_set

    results.append({
        "Year": year,
        "Groundwater_Districts": len(gw_set),
        "Rainfall_Districts": len(rain_set),
        "APY_Districts": len(apy_set),
        "Common_Districts": len(common)
    })

yearly_overlap = pd.DataFrame(results)

print("\nYear-wise common district coverage:")
print(yearly_overlap.to_string(index=False))

yearly_overlap.to_csv(
    OUTPUT_DIR / "yearly_district_overlap.csv",
    index=False
)

print("\nYear-wise overlap report saved successfully.")

# ============================================================
# APY CROP DISTRIBUTION ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("APY CROP DISTRIBUTION ANALYSIS: 2017–2021")
print("=" * 65)

apy = pd.read_csv(
    FILES["APY"],
    usecols=[
        "year",
        "state_name",
        "district_name",
        "crop_name",
        "crop_type",
        "season",
        "area",
        "production",
        "production_unit",
        "yield",
        "yield_unit"
    ],
    low_memory=False
)

# Extract agricultural starting year
apy["start_year"] = pd.to_numeric(
    apy["year"].astype(str).str.extract(r"(\d{4})")[0],
    errors="coerce"
)

# Filter selected agricultural years
apy = apy[
    apy["start_year"].between(2017, 2021)
].copy()

# 1. Total unique crops
print("\nTotal unique crops:")
print(apy["crop_name"].nunique())

# 2. Most frequently recorded crops
print("\nTop 20 crops by number of records:")
print(
    apy["crop_name"]
    .value_counts()
    .head(20)
)

# 3. Crop types
print("\nCrop type distribution:")
print(apy["crop_type"].value_counts(dropna=False))

# 4. Agricultural seasons
print("\nSeason distribution:")
print(apy["season"].value_counts(dropna=False))

# 5. Yield units
print("\nYield units:")
print(apy["yield_unit"].value_counts(dropna=False))

# 6. Production units
print("\nProduction units:")
print(apy["production_unit"].value_counts(dropna=False))

# 7. Save crop distribution
crop_summary = (
    apy.groupby("crop_name")
    .agg(
        Record_Count=("crop_name", "size"),
        Average_Yield=("yield", "mean"),
        Yield_Unit_Count=("yield_unit", "nunique")
    )
    .sort_values("Record_Count", ascending=False)
)

crop_summary.to_csv(
    OUTPUT_DIR / "apy_crop_distribution.csv"
)

print("\nCrop distribution report saved successfully.")
# ============================================================
# APY YIELD AND SEASON VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("APY YIELD AND SEASON VALIDATION")
print("=" * 65)

apy = pd.read_csv(
    FILES["APY"],
    usecols=[
        "year",
        "state_code",
        "district_code",
        "crop_name",
        "season",
        "area",
        "production",
        "yield",
        "yield_unit"
    ],
    low_memory=False
)

# Extract agricultural starting year
apy["start_year"] = pd.to_numeric(
    apy["year"].astype(str).str.extract(r"(\d{4})")[0],
    errors="coerce"
)

# Select 2017–2021
apy = apy[
    apy["start_year"].between(2017, 2021)
].copy()

# Convert numeric columns
for column in ["area", "production", "yield"]:
    apy[column] = pd.to_numeric(
        apy[column],
        errors="coerce"
    )

# 1. Season-wise yield statistics
print("\nSEASON-WISE YIELD STATISTICS")

print(
    apy.groupby("season")["yield"].agg(
        ["count", "mean", "median", "min", "max"]
    ).round(3)
)

# 2. Total season records
print("\nTOTAL SEASON RECORDS")

total_season = apy[
    apy["season"].str.strip().str.lower() == "total"
]

print("Total season rows:", len(total_season))
print("Unique crops:", total_season["crop_name"].nunique())
print("Unique districts:", total_season["district_code"].nunique())

# 3. Yield validation
print("\nYIELD VALIDATION")

print("Missing yields:", apy["yield"].isna().sum())
print("Zero yields:", (apy["yield"] == 0).sum())
print("Negative yields:", (apy["yield"] < 0).sum())

print("\nOverall yield statistics:")
print(apy["yield"].describe().round(3))

# 4. Crop-wise yield statistics
crop_yield = (
    apy.groupby("crop_name")["yield"]
    .agg(
        Record_Count="count",
        Mean_Yield="mean",
        Median_Yield="median",
        Minimum_Yield="min",
        Maximum_Yield="max"
    )
    .sort_values("Record_Count", ascending=False)
    .round(3)
)

print("\nCROP-WISE YIELD STATISTICS:")
print(crop_yield.to_string())

# 5. Save results
crop_yield.to_csv(
    OUTPUT_DIR / "crop_yield_statistics.csv"
)

print("\nYield validation completed.")

# ============================================================
# EXTREME YIELD INVESTIGATION
# ============================================================

print("\n" + "=" * 65)
print("EXTREME YIELD INVESTIGATION")
print("=" * 65)

apy = pd.read_csv(
    FILES["APY"],
    usecols=[
        "year",
        "state_name",
        "district_name",
        "crop_name",
        "season",
        "area",
        "area_unit",
        "production",
        "production_unit",
        "yield",
        "yield_unit"
    ],
    low_memory=False
)

# Extract agricultural starting year
apy["start_year"] = pd.to_numeric(
    apy["year"].astype(str).str.extract(r"(\d{4})")[0],
    errors="coerce"
)

apy = apy[
    apy["start_year"].between(2017, 2021)
].copy()

# Convert numeric columns
for column in ["area", "production", "yield"]:
    apy[column] = pd.to_numeric(
        apy[column],
        errors="coerce"
    )

# 1. Highest 30 yield observations
print("\nTOP 30 HIGHEST YIELD RECORDS:")

highest_yields = apy.nlargest(30, "yield")[
    [
        "year",
        "state_name",
        "district_name",
        "crop_name",
        "season",
        "area",
        "area_unit",
        "production",
        "production_unit",
        "yield",
        "yield_unit"
    ]
]

print(highest_yields.to_string(index=False))

# 2. Yield outlier counts by season using IQR
print("\nOUTLIER COUNTS BY SEASON:")

for season, group in apy.groupby("season"):

    q1 = group["yield"].quantile(0.25)
    q3 = group["yield"].quantile(0.75)

    iqr = q3 - q1

    upper_limit = q3 + 1.5 * iqr
    lower_limit = q1 - 1.5 * iqr

    outliers = group[
        (group["yield"] < lower_limit) |
        (group["yield"] > upper_limit)
    ]

    print(
        f"{season}: {len(outliers)} outliers "
        f"out of {len(group)} records"
    )

# 3. Compare production / area with recorded yield
apy["calculated_yield"] = (
    apy["production"] / apy["area"].replace(0, float("nan"))
)

apy["yield_difference"] = (
    apy["yield"] - apy["calculated_yield"]
).abs()

print("\nLARGEST DIFFERENCES BETWEEN RECORDED AND CALCULATED YIELD:")

comparison = apy.nlargest(20, "yield_difference")[
    [
        "crop_name",
        "season",
        "area",
        "production",
        "yield",
        "calculated_yield",
        "yield_difference"
    ]
]

print(comparison.to_string(index=False))

# 4. Save extreme records for inspection
highest_yields.to_csv(
    OUTPUT_DIR / "extreme_yield_records.csv",
    index=False
)

comparison.to_csv(
    OUTPUT_DIR / "yield_calculation_comparison.csv",
    index=False
)

print("\nExtreme yield investigation saved successfully.")

# ============================================================
# APY SEASON DUPLICATION INVESTIGATION
# ============================================================

print("\n" + "=" * 65)
print("APY SEASON DUPLICATION INVESTIGATION")
print("=" * 65)

apy = pd.read_csv(
    FILES["APY"],
    usecols=[
        "year",
        "state_code",
        "district_code",
        "crop_name",
        "season",
        "area",
        "production",
        "yield"
    ],
    low_memory=False
)

# Extract agricultural starting year
apy["start_year"] = pd.to_numeric(
    apy["year"].astype(str).str.extract(r"(\d{4})")[0],
    errors="coerce"
)

apy = apy[
    apy["start_year"].between(2017, 2021)
].copy()

# Remove extra spaces from season names
apy["season"] = apy["season"].astype(str).str.strip()

# 1. Check duplicate natural-key combinations
key_columns = [
    "start_year",
    "state_code",
    "district_code",
    "crop_name",
    "season"
]

duplicate_count = apy.duplicated(
    subset=key_columns,
    keep=False
).sum()

print("\nDuplicate records using natural keys:", duplicate_count)

# 2. Check how often Total and other seasons coexist
group_columns = [
    "start_year",
    "state_code",
    "district_code",
    "crop_name"
]

season_combinations = (
    apy.groupby(group_columns)["season"]
    .agg(lambda x: sorted(set(x)))
    .reset_index(name="Seasons")
)

season_combinations["Has_Total"] = (
    season_combinations["Seasons"]
    .apply(lambda x: "Total" in x)
)

total_groups = season_combinations[
    season_combinations["Has_Total"]
]

print("\nCrop-district-year groups containing Total:")
print(len(total_groups))

print("\nSeason combinations among groups containing Total:")
print(
    total_groups["Seasons"]
    .astype(str)
    .value_counts()
    .head(20)
)

# 3. Show examples where Total coexists with other seasons
examples = apy.merge(
    total_groups[group_columns],
    on=group_columns,
    how="inner"
)

print("\nExample records containing Total:")
print(
    examples.sort_values(
        group_columns + ["season"]
    ).head(40).to_string(index=False)
)

# 4. Save results
season_combinations.to_csv(
    OUTPUT_DIR / "apy_season_combinations.csv",
    index=False
)

examples.to_csv(
    OUTPUT_DIR / "apy_total_season_examples.csv",
    index=False
)

print("\nSeason duplication investigation saved successfully.")

# ============================================================
# VERIFY TOTAL SEASON AGGREGATION
# ============================================================

print("\n" + "=" * 65)
print("VERIFYING TOTAL SEASON AGGREGATION")
print("=" * 65)

# Separate aggregate and individual season records
total_rows = apy[
    apy["season"].str.strip().str.lower() == "total"
].copy()

seasonal_rows = apy[
    apy["season"].str.strip().str.lower() != "total"
].copy()

# Group individual seasons by crop, district and year
group_columns = [
    "start_year",
    "state_code",
    "district_code",
    "crop_name"
]

seasonal_sums = (
    seasonal_rows.groupby(group_columns)
    .agg(
        Seasonal_Area=("area", "sum"),
        Seasonal_Production=("production", "sum")
    )
    .reset_index()
)

# Select Total records
total_comparison = total_rows[
    group_columns + ["area", "production"]
].merge(
    seasonal_sums,
    on=group_columns,
    how="left"
)

# Compare aggregate values
total_comparison["Area_Match"] = (
    (total_comparison["area"] -
     total_comparison["Seasonal_Area"]).abs() < 0.01
)

total_comparison["Production_Match"] = (
    (total_comparison["production"] -
     total_comparison["Seasonal_Production"]).abs() < 0.01
)

print("\nTotal records:", len(total_comparison))

print(
    "Area matches:",
    total_comparison["Area_Match"].sum()
)

print(
    "Production matches:",
    total_comparison["Production_Match"].sum()
)

print(
    "Both match:",
    (
        total_comparison["Area_Match"] &
        total_comparison["Production_Match"]
    ).sum()
)

print("\nRecords where aggregate values do not match:")

mismatches = total_comparison[
    ~(
        total_comparison["Area_Match"] &
        total_comparison["Production_Match"]
    )
]

print(mismatches.head(20).to_string(index=False))

total_comparison.to_csv(
    OUTPUT_DIR / "total_season_verification.csv",
    index=False
)

print("\nTotal season verification saved successfully.")

# ============================================================
# WHOLE YEAR SEASON INVESTIGATION
# ============================================================

print("\n" + "=" * 65)
print("WHOLE YEAR SEASON INVESTIGATION")
print("=" * 65)

# Select Whole Year records
whole_year = apy[
    apy["season"].str.strip().str.lower() == "whole year"
].copy()

# 1. Basic statistics
print("\nWhole Year record count:", len(whole_year))

print("\nCrop distribution:")
print(whole_year["crop_name"].value_counts().head(20))

print("\nYield statistics:")
print(whole_year["yield"].describe())

# 2. Crop-wise Whole Year statistics
print("\nCrop-wise Whole Year yield statistics:")

print(
    whole_year.groupby("crop_name")["yield"]
    .agg(["count", "mean", "median", "min", "max"])
    .sort_values("count", ascending=False)
    .round(3)
    .to_string()
)

# 3. Check whether Whole Year coexists
# with other seasons for the same crop/district/year

group_columns = [
    "start_year",
    "state_code",
    "district_code",
    "crop_name"
]

season_groups = (
    apy.groupby(group_columns)["season"]
    .agg(lambda x: sorted(set(x)))
    .reset_index(name="Seasons")
)

whole_year_groups = season_groups[
    season_groups["Seasons"].apply(
        lambda x: "Whole Year" in x
    )
]

print(
    "\nCrop-district-year groups containing Whole Year:",
    len(whole_year_groups)
)

print("\nSeason combinations:")
print(
    whole_year_groups["Seasons"]
    .astype(str)
    .value_counts()
    .head(20)
)

# 4. Save records
whole_year.to_csv(
    OUTPUT_DIR / "whole_year_records.csv",
    index=False
)

whole_year_groups.to_csv(
    OUTPUT_DIR / "whole_year_season_combinations.csv",
    index=False
)

print("\nWhole Year investigation saved successfully.")
# ============================================================
# COCONUT YIELD INVESTIGATION
# ============================================================

print("\n" + "=" * 65)
print("COCONUT YIELD INVESTIGATION")
print("=" * 65)

coconut = pd.read_csv(
    FILES["APY"],
    usecols=[
        "year",
        "state_name",
        "district_name",
        "crop_name",
        "season",
        "area",
        "production",
        "yield",
        "area_unit",
        "production_unit",
        "yield_unit"
    ],
    low_memory=False
)

# Extract agricultural starting year
coconut["start_year"] = pd.to_numeric(
    coconut["year"].astype(str).str.extract(r"(\d{4})")[0],
    errors="coerce"
)

# Filter 2017–2021
coconut = coconut[
    coconut["start_year"].between(2017, 2021)
].copy()

# Select Coconut
coconut = coconut[
    coconut["crop_name"].str.strip().str.lower() == "coconut"
].copy()

# Convert numerical columns
for column in ["area", "production", "yield"]:
    coconut[column] = pd.to_numeric(
        coconut[column],
        errors="coerce"
    )

# Sort by yield
coconut = coconut.sort_values(
    "yield",
    ascending=False
)

print("\nTotal coconut records:", len(coconut))

print("\nCoconut yield statistics:")
print(coconut["yield"].describe())

print("\nCoconut records by season:")
print(coconut["season"].value_counts())

columns = [
    "year",
    "state_name",
    "district_name",
    "season",
    "area",
    "production",
    "yield",
    "area_unit",
    "production_unit",
    "yield_unit"
]

print("\nTop 20 coconut yield records:")
print(
    coconut[columns].head(20).to_string(index=False)
)

print("\nCoconut yield by season:")
print(
    coconut.groupby("season")["yield"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(3)
)

# Save results
coconut[columns].to_csv(
    OUTPUT_DIR / "coconut_yield_investigation.csv",
    index=False
)

print("\nCoconut investigation saved successfully.")

# ============================================================
# CROP-WISE AREA, PRODUCTION AND YIELD VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("CROP-WISE AREA, PRODUCTION AND YIELD VALIDATION")
print("=" * 65)

apy_check = pd.read_csv(
    FILES["APY"],
    usecols=[
        "year",
        "crop_name",
        "season",
        "area",
        "production",
        "yield",
        "area_unit",
        "production_unit",
        "yield_unit"
    ],
    low_memory=False
)

# Extract agricultural starting year
apy_check["start_year"] = pd.to_numeric(
    apy_check["year"].astype(str).str.extract(r"(\d{4})")[0],
    errors="coerce"
)

# Filter 2017–2021
apy_check = apy_check[
    apy_check["start_year"].between(2017, 2021)
].copy()

# Convert numeric columns
for column in ["area", "production", "yield"]:
    apy_check[column] = pd.to_numeric(
        apy_check[column],
        errors="coerce"
    )

# Exclude Total for this crop-wise seasonal comparison
apy_check = apy_check[
    apy_check["season"].str.strip().str.lower() != "total"
].copy()

# Crop-wise statistics
crop_validation = (
    apy_check.groupby("crop_name")
    .agg(
        Records=("yield", "count"),
        Mean_Area=("area", "mean"),
        Maximum_Area=("area", "max"),
        Mean_Production=("production", "mean"),
        Maximum_Production=("production", "max"),
        Mean_Yield=("yield", "mean"),
        Median_Yield=("yield", "median"),
        Maximum_Yield=("yield", "max")
    )
    .sort_values("Maximum_Yield", ascending=False)
    .round(3)
)

print("\nTop 20 crops by maximum yield:")
print(crop_validation.head(20).to_string())

# Save results
crop_validation.to_csv(
    OUTPUT_DIR / "crop_area_production_yield_validation.csv"
)

print("\nCrop-wise validation saved successfully.")

print("\n" + "=" * 65)
print("INVESTIGATION OF EXTREME CROP YIELD RECORDS")
print("=" * 65)

apy = pd.read_csv(
    FILES["APY"],
    usecols=[
        "year", "state_name", "district_name",
        "crop_name", "season", "area",
        "production", "yield",
        "area_unit", "production_unit", "yield_unit"
    ]
)

apy["start_year"] = (
    apy["year"].astype(str).str[:4].astype(int)
)

apy = apy[
    apy["start_year"].between(2017, 2021)
].copy()

# Exclude Total to avoid aggregate-season duplication
apy = apy[apy["season"] != "Total"].copy()

crops_to_check = [
    "Coconut",
    "Onion",
    "Sannhamp",
    "Turmeric",
    "Sugarcane"
]

for crop in crops_to_check:

    crop_data = apy[apy["crop_name"] == crop]

    print(f"\n{'-' * 65}")
    print("CROP:", crop)
    print(f"{'-' * 65}")

    print(
        crop_data.nlargest(5, "yield")[
            [
                "year", "state_name", "district_name",
                "season", "area", "production",
                "yield", "area_unit",
                "production_unit", "yield_unit"
            ]
        ].to_string(index=False)
    )
    
    print("\n" + "=" * 65)
print("APY MISSING PRODUCTION AND ZERO-YIELD ANALYSIS")
print("=" * 65)

# Missing production
missing_production = apy[apy["production"].isna()]

print("\nMissing production records:", len(missing_production))

print(
    missing_production[
        [
            "year", "state_name", "district_name",
            "crop_name", "season", "area", "yield"
        ]
    ].head(10).to_string(index=False)
)

# Zero-yield records
zero_yield = apy[apy["yield"] == 0]

print("\nZero-yield records:", len(zero_yield))

print("\nZero-yield distribution by season:")
print(zero_yield["season"].value_counts())

print("\nTop crops with zero yield:")
print(zero_yield["crop_name"].value_counts().head(15))

print("\nZero-yield records with positive production:")
print(
    len(
        zero_yield[
            zero_yield["production"] > 0
        ]
    )
)

print("\n" + "=" * 65)
print("ZERO-YIELD AND PRODUCTION CROSS-VALIDATION")
print("=" * 65)

zero_yield = apy[apy["yield"] == 0].copy()

print("\nZero-yield records:", len(zero_yield))

print("\nProduction status among zero-yield records:")

production_status = zero_yield["production"].apply(
    lambda x:
        "Missing production" if pd.isna(x)
        else "Zero production" if x == 0
        else "Positive production" if x > 0
        else "Negative production"
)

print(production_status.value_counts(dropna=False))

print("\nDetailed positive-production zero-yield record:")

print(
    zero_yield[
        zero_yield["production"] > 0
    ][
        [
            "year", "state_name", "district_name",
            "crop_name", "season", "area",
            "production", "yield"
        ]
    ].to_string(index=False)
)
      
      # ==========================================
# SAVE CLEANED APY MODELLING DATASET
# ==========================================

apy_model = apy.dropna(
    subset=["production"]
).copy()

# Keep valid non-negative observations
apy_model = apy_model[
    (apy_model["area"] > 0) &
    (apy_model["production"] >= 0) &
    (apy_model["yield"] >= 0)
].copy()

# Save cleaned dataset
apy_output = OUTPUT_DIR / "apy_cleaned.csv"

apy_model.to_csv(apy_output, index=False)

print("\nAPY CLEANING COMPLETED")
print("Original selected records:", len(apy))
print("Final modelling records:", len(apy_model))
print("Saved to:", apy_output)
print("Missing production:", apy_model["production"].isna().sum())
print("Zero-yield records retained:", (apy_model["yield"] == 0).sum())

# ==========================================
# RAINFALL AGGREGATION
# ==========================================

print("\nAGGREGATING RAINFALL DATA")

rainfall = pd.read_csv(
    FILES["Rainfall"],
    usecols=[
        "date", "state_code", "district_code",
        "actual"
    ],
    parse_dates=["date"]
)

# Retain July 2017 to June 2022
rainfall = rainfall[
    rainfall["date"].between(
        "2017-07-01", "2022-06-30"
    )
].copy()

# Agricultural year starting year
rainfall["agri_year"] = (
    rainfall["date"].dt.year
    - (rainfall["date"].dt.month < 7).astype(int)
)

# District-level agricultural-year aggregation
rainfall_yearly = (
    rainfall.groupby(
        ["state_code", "district_code", "agri_year"]
    )
    .agg(
        rainfall_total=("actual", lambda x: x.sum(min_count=1)),
        rainfall_mean=("actual", "mean"),
        rainfall_days=("actual", "count")
    )
    .reset_index()
)

rainfall_output = OUTPUT_DIR / "rainfall_yearly.csv"
rainfall_yearly.to_csv(rainfall_output, index=False)

print("Rainfall aggregation completed")
print("Aggregated records:", len(rainfall_yearly))
print("Saved to:", rainfall_output)
print(rainfall_yearly.head())

# ==========================================
# GROUNDWATER AGGREGATION
# ==========================================

print("\nAGGREGATING GROUNDWATER DATA")

groundwater = pd.read_csv(
    FILES["Groundwater"],
    usecols=[
        "date",
        "state_code",
        "district_code",
        "station_name",
        "currentlevel",
        "level_diff"
    ],
    parse_dates=["date"]
)

# Retain July 2017 to June 2022
groundwater = groundwater[
    groundwater["date"].between(
        "2017-07-01", "2022-06-30"
    )
].copy()

# Assign agricultural year
groundwater["agri_year"] = (
    groundwater["date"].dt.year
    - (groundwater["date"].dt.month < 7).astype(int)
)

# District-level agricultural-year aggregation
groundwater_yearly = (
    groundwater.groupby(
        ["state_code", "district_code", "agri_year"]
    )
    .agg(
        groundwater_median=("currentlevel", "median"),
        groundwater_mean=("currentlevel", "mean"),
        level_change_mean=("level_diff", "mean"),
        station_count=("station_name", "nunique"),
        groundwater_records=("currentlevel", "count")
    )
    .reset_index()
)

groundwater_output = OUTPUT_DIR / "groundwater_yearly.csv"

groundwater_yearly.to_csv(
    groundwater_output,
    index=False
)

print("Groundwater aggregation completed")
print("Aggregated records:", len(groundwater_yearly))
print("Saved to:", groundwater_output)
print(groundwater_yearly.head())

# ============================================================
# INTEGRATE APY, RAINFALL AND GROUNDWATER DATA
# ============================================================

print("\n" + "=" * 65)
print("INTEGRATING ALL THREE DATASETS")
print("=" * 65)

# 1. Reload APY with all required columns
apy_model = pd.read_csv(
    FILES["APY"],
    usecols=[
        "year",
        "state_code",
        "district_code",
        "state_name",
        "district_name",
        "crop_name",
        "season",
        "area",
        "production",
        "yield",
        "area_unit",
        "production_unit",
        "yield_unit"
    ]
)

# 2. Select agricultural years 2017-2021
apy_model["agri_year"] = (
    apy_model["year"].astype(str).str[:4].astype(int)
)

apy_model = apy_model[
    apy_model["agri_year"].between(2017, 2021)
].copy()

# 3. Apply APY cleaning rules
apy_model = apy_model[
    (apy_model["season"] != "Total") &
    (apy_model["production"].notna()) &
    (apy_model["area"] > 0) &
    (apy_model["production"] >= 0) &
    (apy_model["yield"] >= 0)
].copy()

# 4. Load aggregated rainfall and groundwater
rainfall_yearly = pd.read_csv(
    OUTPUT_DIR / "rainfall_yearly.csv"
)

groundwater_yearly = pd.read_csv(
    OUTPUT_DIR / "groundwater_yearly.csv"
)

# 5. Ensure join-key data types match
join_keys = ["state_code", "district_code", "agri_year"]

for df in [apy_model, rainfall_yearly, groundwater_yearly]:
    for col in join_keys:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove records with invalid join keys
apy_model = apy_model.dropna(subset=join_keys)

rainfall_yearly = rainfall_yearly.dropna(subset=join_keys)
groundwater_yearly = groundwater_yearly.dropna(subset=join_keys)

# Convert join keys to integers
for df in [apy_model, rainfall_yearly, groundwater_yearly]:
    df[join_keys] = df[join_keys].astype(int)

# 6. Check duplicate keys in aggregated datasets
print("\nRainfall duplicate join keys:",
      rainfall_yearly.duplicated(join_keys).sum())

print("Groundwater duplicate join keys:",
      groundwater_yearly.duplicated(join_keys).sum())

# 7. Merge APY with rainfall
integrated = apy_model.merge(
    rainfall_yearly,
    on=join_keys,
    how="left",
    validate="many_to_one"
)

# 8. Merge groundwater
integrated = integrated.merge(
    groundwater_yearly,
    on=join_keys,
    how="left",
    validate="many_to_one"
)

# 9. Save integrated dataset
integrated_output = (
    OUTPUT_DIR / "integrated_agriculture_dataset.csv"
)

integrated.to_csv(integrated_output, index=False)

# 10. Display integration results
print("\n" + "=" * 65)
print("INTEGRATION COMPLETED")
print("=" * 65)

print("APY records:", len(apy_model))
print("Integrated records:", len(integrated))
print("Final dataset shape:", integrated.shape)

print("\nMissing values in environmental features:")

environmental_features = [
    "rainfall_total",
    "rainfall_mean",
    "rainfall_days",
    "groundwater_median",
    "groundwater_mean",
    "level_change_mean",
    "station_count",
    "groundwater_records"
]

print(integrated[environmental_features].isnull().sum())

print("\nSaved to:", integrated_output)

print("\nFirst five records:")
print(integrated.head())