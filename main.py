# ================================================================
# AGRICULTURAL PRODUCTIVITY PREDICTION
# INT234 - PREDICTIVE ANALYTICS
#
# Project:
# Agricultural Productivity Prediction Using Rainfall
# and Groundwater Indicators Across Indian Districts
#
# Models:
# 1. Linear Regression
# 2. Decision Tree Regression
# 3. Random Forest Regression
# 4. Gradient Boosting Regression
# 5. Linear Support Vector Regression
# ================================================================


# ================================================================
# SECTION 1: IMPORT LIBRARIES
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



# ================================================================
# SECTION 2: PROJECT PATHS AND DATA LOADING
# ================================================================

BASE_DIR = Path(
    r"C:\Users\NajHi\OneDrive\Desktop\lpu\sem 5\INT234 pred analysis\CA2 proj"
)

DATA_DIR = BASE_DIR / "processed_data"

OUTPUT_DIR = BASE_DIR / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graphs"
RESULT_DIR = OUTPUT_DIR / "model_results"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


print("\n" + "=" * 75)
print("AGRICULTURAL PRODUCTIVITY PREDICTION")
print("=" * 75)


data = pd.read_csv(
    DATA_DIR / "integrated_agriculture_dataset.csv"
)

print("\nDataset successfully loaded.")

print("Dataset shape:", data.shape)

print("\nDataset information:")
print(data.info())

print("\nFirst five records:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())


# ================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
# ================================================================

print("\n" + "=" * 75)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 75)


# ------------------------------------------------
# 3.1 TARGET VARIABLE DISTRIBUTION
# ------------------------------------------------

print("\nYield statistics:")
print(data["yield"].describe())

print("\nCrop categories:", data["crop_name"].nunique())
print("Season categories:", data["season"].nunique())


# Yield distribution
plt.figure(figsize=(10, 6))

plt.hist(
    data["yield"],
    bins=60,
    edgecolor="black"
)

plt.title("Distribution of Agricultural Yield")
plt.xlabel("Yield (Tonnes per Hectare)")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "01_yield_distribution.png",
    dpi=300
)

plt.show()


# Log-transformed yield distribution
plt.figure(figsize=(10, 6))

plt.hist(
    np.log1p(data["yield"]),
    bins=60,
    edgecolor="black"
)

plt.title("Log-Transformed Agricultural Yield Distribution")
plt.xlabel("Log(1 + Yield)")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "02_log_yield_distribution.png",
    dpi=300
)

plt.show()


# ------------------------------------------------
# 3.2 CROP-WISE YIELD ANALYSIS
# ------------------------------------------------

crop_yield = (
    data.groupby("crop_name")["yield"]
    .median()
    .sort_values(ascending=False)
)

top_crops = crop_yield.head(15).sort_values()

plt.figure(figsize=(11, 8))

plt.barh(
    top_crops.index,
    top_crops.values
)

plt.title("Top 15 Crops by Median Yield")
plt.xlabel("Median Yield (Tonnes per Hectare)")
plt.ylabel("Crop")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "03_crop_yield_comparison.png",
    dpi=300
)

plt.show()


# ------------------------------------------------
# 3.3 SEASON-WISE YIELD ANALYSIS
# ------------------------------------------------

season_yield = (
    data.groupby("season")["yield"]
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

plt.bar(
    season_yield.index,
    season_yield.values
)

plt.title("Season-Wise Median Agricultural Yield")
plt.xlabel("Agricultural Season")
plt.ylabel("Median Yield (Tonnes per Hectare)")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "04_season_yield_comparison.png",
    dpi=300
)

plt.show()


# ------------------------------------------------
# 3.4 YEAR-WISE YIELD TREND
# ------------------------------------------------

year_yield = (
    data.groupby("agri_year")["yield"]
    .median()
)

plt.figure(figsize=(9, 6))

plt.plot(
    year_yield.index,
    year_yield.values,
    marker="o",
    linewidth=2
)

plt.title("Agricultural Yield Trend Across Years")
plt.xlabel("Agricultural Year")
plt.ylabel("Median Yield (Tonnes per Hectare)")

plt.xticks(year_yield.index)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "05_yearly_yield_trend.png",
    dpi=300
)

plt.show()


# ------------------------------------------------
# 3.5 CORRELATION ANALYSIS
# ------------------------------------------------

numeric_columns = [
    "agri_year",
    "rainfall_total",
    "rainfall_mean",
    "rainfall_days",
    "groundwater_median",
    "groundwater_mean",
    "level_change_mean",
    "station_count",
    "groundwater_records",
    "yield"
]

correlation = data[numeric_columns].corr()

plt.figure(figsize=(12, 9))

image = plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="auto"
)

plt.colorbar(image, label="Correlation")

plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=60,
    ha="right"
)

plt.yticks(
    range(len(numeric_columns)),
    numeric_columns
)

plt.title("Correlation Heatmap of Numerical Features")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "06_correlation_heatmap.png",
    dpi=300
)

plt.show()


# ------------------------------------------------
# 3.6 RAINFALL AND YIELD RELATIONSHIP
# ------------------------------------------------

rain_data = data[
    data["rainfall_total"].notna()
].sample(
    n=min(5000, data["rainfall_total"].notna().sum()),
    random_state=42
)

plt.figure(figsize=(9, 6))

plt.scatter(
    rain_data["rainfall_total"],
    rain_data["yield"],
    alpha=0.35
)

plt.title("Relationship Between Rainfall and Agricultural Yield")
plt.xlabel("Total Rainfall")
plt.ylabel("Yield (Tonnes per Hectare)")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "07_rainfall_vs_yield.png",
    dpi=300
)

plt.show()


# ------------------------------------------------
# 3.7 GROUNDWATER AND YIELD RELATIONSHIP
# ------------------------------------------------

ground_data = data[
    data["groundwater_median"].notna()
].sample(
    n=min(
        5000,
        data["groundwater_median"].notna().sum()
    ),
    random_state=42
)

plt.figure(figsize=(9, 6))

plt.scatter(
    ground_data["groundwater_median"],
    ground_data["yield"],
    alpha=0.35
)

plt.title("Relationship Between Groundwater and Agricultural Yield")
plt.xlabel("Median Groundwater Level")
plt.ylabel("Yield (Tonnes per Hectare)")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "08_groundwater_vs_yield.png",
    dpi=300
)

plt.show()


# Save EDA summary
eda_summary = pd.DataFrame({
    "Metric": [
        "Total Records",
        "Number of Crops",
        "Number of Seasons",
        "Mean Yield",
        "Median Yield",
        "Maximum Yield",
        "Minimum Yield"
    ],
    "Value": [
        len(data),
        data["crop_name"].nunique(),
        data["season"].nunique(),
        data["yield"].mean(),
        data["yield"].median(),
        data["yield"].max(),
        data["yield"].min()
    ]
})

eda_summary.to_csv(
    RESULT_DIR / "eda_summary.csv",
    index=False
)


# ================================================================
# SECTION 4: FEATURE SELECTION AND PREPROCESSING
# ================================================================

print("\n" + "=" * 75)
print("FEATURE SELECTION AND PREPROCESSING")
print("=" * 75)


# Categorical predictors
categorical_features = [
    "state_code",
    "district_code",
    "crop_name",
    "season"
]


# Numerical predictors
numerical_features = [
    "agri_year",
    "rainfall_total",
    "rainfall_mean",
    "rainfall_days",
    "groundwater_median",
    "groundwater_mean",
    "level_change_mean",
    "station_count",
    "groundwater_records"
]


features = categorical_features + numerical_features

X = data[features].copy()

y = data["yield"].copy()


# Convert geographic identifiers to categorical strings
for column in categorical_features:
    X[column] = X[column].astype(str)


# Log transformation of target variable
y_log = np.log1p(y)


# ------------------------------------------------
# 4.1 CHRONOLOGICAL TRAIN-TEST SPLIT
# ------------------------------------------------

train_mask = data["agri_year"] < 2021
test_mask = data["agri_year"] == 2021

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y_log.loc[train_mask]
y_test = y.loc[test_mask]


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Training years:",
    sorted(data.loc[train_mask, "agri_year"].unique())
)

print(
    "Testing year:",
    sorted(data.loc[test_mask, "agri_year"].unique())
)


# ------------------------------------------------
# 4.2 NUMERICAL PREPROCESSING
# ------------------------------------------------

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median",
            add_indicator=True
        )
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# ------------------------------------------------
# 4.3 CATEGORICAL PREPROCESSING
# ------------------------------------------------

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


# ------------------------------------------------
# 4.4 COMBINE PREPROCESSING
# ------------------------------------------------

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numerical_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


print("\nPreprocessing pipeline created successfully.")


# ================================================================
# SECTION 5: MACHINE LEARNING MODEL DEVELOPMENT
# ================================================================

print("\n" + "=" * 75)
print("MACHINE LEARNING MODEL DEVELOPMENT")
print("=" * 75)


# ------------------------------------------------
# 5.1 LINEAR REGRESSION
# ------------------------------------------------

linear_model = LinearRegression()


# ------------------------------------------------
# 5.2 DECISION TREE REGRESSION
# ------------------------------------------------

decision_tree_model = DecisionTreeRegressor(
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)


# ------------------------------------------------
# 5.3 RANDOM FOREST REGRESSION
# ------------------------------------------------

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42
)


# ------------------------------------------------
# 5.4 GRADIENT BOOSTING REGRESSION
# ------------------------------------------------

gradient_boosting_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)


# ------------------------------------------------
# 5.5 LINEAR SUPPORT VECTOR REGRESSION
# ------------------------------------------------

svr_model = LinearSVR(
    C=1.0,
    epsilon=0.0,
    max_iter=30000,
    tol=1e-3,
    random_state=42
)


# Store all models
models = {
    "Linear Regression": linear_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "Gradient Boosting": gradient_boosting_model,
    "Linear SVR": svr_model
}


# ================================================================
# SECTION 6: MODEL TRAINING AND EVALUATION
# ================================================================

print("\n" + "=" * 75)
print("TRAINING AND EVALUATING MODELS")
print("=" * 75)


results = []

predictions = {}

trained_models = {}

feature_importance_data = {}


for name, model in models.items():

    print("\n" + "-" * 65)
    print("TRAINING MODEL:", name)
    print("-" * 65)


    # Create independent preprocessing pipeline for each model
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


    # Train model
    pipeline.fit(
        X_train,
        y_train
    )


    # Predict log-transformed yield
    predicted_log = pipeline.predict(X_test)


    # Convert predictions to original scale
    predicted_yield = np.maximum(
        0,
        np.expm1(predicted_log)
    )


    actual_yield = y_test.to_numpy()


    # ------------------------------------------------
    # EVALUATION METRICS
    # ------------------------------------------------

    mae = mean_absolute_error(
        actual_yield,
        predicted_yield
    )

    mse = mean_squared_error(
        actual_yield,
        predicted_yield
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual_yield,
        predicted_yield
    )


    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })


    predictions[name] = predicted_yield

    trained_models[name] = pipeline


    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))


    # ------------------------------------------------
    # FEATURE IMPORTANCE / COEFFICIENTS
    # ------------------------------------------------

    fitted_model = pipeline.named_steps["model"]

    fitted_preprocessor = pipeline.named_steps["preprocessor"]

    transformed_feature_names = (
        fitted_preprocessor.get_feature_names_out()
    )


    if hasattr(fitted_model, "feature_importances_"):

        importance_values = fitted_model.feature_importances_

        feature_importance_data[name] = pd.Series(
            importance_values,
            index=transformed_feature_names
        ).sort_values(ascending=False)


    elif hasattr(fitted_model, "coef_"):

        coefficients = np.ravel(fitted_model.coef_)

        feature_importance_data[name] = pd.Series(
            np.abs(coefficients),
            index=transformed_feature_names
        ).sort_values(ascending=False)


print("\nAll five models have been trained.")


# ================================================================
# SECTION 7: MODEL COMPARISON
# ================================================================

print("\n" + "=" * 75)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 75)


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE"
).reset_index(drop=True)


print("\nFinal model comparison:")
print(results_df.to_string(index=False))


# Save comparison
results_df.to_csv(
    RESULT_DIR / "model_comparison.csv",
    index=False
)


# ------------------------------------------------
# 7.1 MAE, RMSE AND R2 COMPARISON
# ------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)

metrics = [
    ("MAE", "Mean Absolute Error"),
    ("RMSE", "Root Mean Squared Error"),
    ("R2", "R-Squared Score")
]

for ax, (metric, title) in zip(axes, metrics):

    ax.bar(
        results_df["Model"],
        results_df[metric]
    )

    ax.set_title(title)
    ax.set_ylabel(metric)

    ax.tick_params(
        axis="x",
        rotation=45
    )

fig.suptitle(
    "Comparison of Five Regression Models",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "09_model_performance_comparison.png",
    dpi=300
)

plt.show()


# ================================================================
# SECTION 8: PREDICTION VISUALISATIONS
# ================================================================

print("\n" + "=" * 75)
print("GENERATING PREDICTION VISUALISATIONS")
print("=" * 75)


# ------------------------------------------------
# 8.1 ACTUAL VS PREDICTED VALUES
# ------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 12)
)

axes = axes.flatten()

actual = y_test.to_numpy()

for index, (name, predicted) in enumerate(predictions.items()):

    axes[index].scatter(
        actual,
        predicted,
        alpha=0.3
    )

    maximum = max(
        np.max(actual),
        np.max(predicted)
    )

    axes[index].plot(
        [0, maximum],
        [0, maximum],
        linestyle="--"
    )

    axes[index].set_title(name)
    axes[index].set_xlabel("Actual Yield")
    axes[index].set_ylabel("Predicted Yield")

axes[-1].axis("off")

fig.suptitle(
    "Actual vs Predicted Agricultural Yield",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "10_actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ------------------------------------------------
# 8.2 RESIDUAL ANALYSIS OF BEST MODEL
# ------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_predictions = predictions[best_model_name]

residuals = actual - best_predictions


fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)


# Residual scatter plot
axes[0].scatter(
    best_predictions,
    residuals,
    alpha=0.35
)

axes[0].axhline(
    y=0,
    linestyle="--"
)

axes[0].set_title(
    f"Residual Plot - {best_model_name}"
)

axes[0].set_xlabel("Predicted Yield")
axes[0].set_ylabel("Residuals")


# Residual distribution
axes[1].hist(
    residuals,
    bins=50,
    edgecolor="black"
)

axes[1].set_title(
    "Distribution of Prediction Residuals"
)

axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")


plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "11_residual_analysis.png",
    dpi=300
)

plt.show()


# ================================================================
# SECTION 9: FEATURE IMPORTANCE ANALYSIS
# ================================================================

print("\n" + "=" * 75)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 75)


for name, importance in feature_importance_data.items():

    print("\nTop 15 features for:", name)

    print(importance.head(15))

    top_features = importance.head(15).sort_values()

    plt.figure(figsize=(11, 8))

    plt.barh(
        top_features.index,
        top_features.values
    )

    plt.title(
        f"Top 15 Feature Importance - {name}"
    )

    plt.xlabel(
        "Absolute Coefficient / Feature Importance"
    )

    plt.ylabel("Features")

    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")

    plt.savefig(
        GRAPH_DIR / f"feature_importance_{safe_name}.png",
        dpi=300
    )

    plt.show()


# Save feature importance
for name, importance in feature_importance_data.items():

    safe_name = name.lower().replace(" ", "_")

    importance.to_csv(
        RESULT_DIR / f"feature_importance_{safe_name}.csv",
        header=["Importance"]
    )


# ================================================================
# SECTION 10: SAVE PREDICTIONS
# ================================================================

print("\n" + "=" * 75)
print("SAVING MODEL PREDICTIONS")
print("=" * 75)


prediction_df = pd.DataFrame({
    "Actual_Yield": actual
})


for name, predicted in predictions.items():

    prediction_df[name] = predicted


prediction_df.to_csv(
    RESULT_DIR / "model_predictions.csv",
    index=False
)


# Save actual and residual values for the best model
best_result = results_df.iloc[0]

best_model_summary = pd.DataFrame({
    "Actual_Yield": actual,
    "Predicted_Yield": best_predictions,
    "Residual": residuals
})

best_model_summary.to_csv(
    RESULT_DIR / "best_model_predictions.csv",
    index=False
)


# ================================================================
# SECTION 11: FINAL MODEL SUMMARY
# ================================================================

print("\n" + "=" * 75)
print("FINAL PROJECT RESULTS")
print("=" * 75)


print("\nBest Performing Model:", best_model_name)

print("\nBest Model Evaluation:")

print("MAE:", round(best_result["MAE"], 4))

print("MSE:", round(best_result["MSE"], 4))

print("RMSE:", round(best_result["RMSE"], 4))

print("R2 Score:", round(best_result["R2"], 4))


print("\nModel Ranking:")

for index, row in results_df.iterrows():

    print(
        f"{index + 1}. {row['Model']} "
        f"(RMSE: {row['RMSE']:.4f})"
    )


print("\nGraph directory:")
print(GRAPH_DIR)

print("\nModel results directory:")
print(RESULT_DIR)


print("\n" + "=" * 75)
print("AGRICULTURAL PRODUCTIVITY PREDICTION COMPLETED")
print("=" * 75)