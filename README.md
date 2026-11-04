# Agricultural Productivity Prediction Using Rainfall and Groundwater Indicators

A machine learning project that analyses agricultural productivity across Indian districts and estimates crop yield using rainfall, groundwater and agricultural indicators.

## About the Project

Agricultural productivity varies across regions due to differences in environmental conditions, water availability, crop types and seasons.

This project combines three datasets — rainfall, groundwater and agricultural production — to study these variations and develop machine learning regression models for crop yield estimation.

The analysis focuses on five agricultural years, from **2017 to 2021**, and compares five regression algorithms to evaluate their predictive performance.

## Problem Statement

To analyse agricultural productivity across Indian districts and develop machine learning regression models to estimate crop yield using rainfall, groundwater and other relevant agricultural features.

## Objectives

- Collect and analyse rainfall, groundwater and agricultural production datasets.
- Perform data cleaning, preprocessing and feature engineering.
- Aggregate environmental measurements at district and agricultural-year levels.
- Integrate the datasets using geographical and temporal identifiers.
- Perform Exploratory Data Analysis (EDA) to identify agricultural patterns.
- Implement and compare five regression algorithms.
- Evaluate model performance using standard regression metrics.

## Datasets

The project uses three publicly available datasets from the India Data Portal.

| Dataset | Description |
|---|---|
| Groundwater | District-level groundwater measurements |
| Rainfall | Daily district-level rainfall observations |
| APY | Crop-wise Area, Production and Yield statistics |

**Dataset sources:**

1. [Groundwater Dataset](https://ckandev.indiadataportal.com/dataset/groundwater/resource/580a8f6e-3d86-4ca7-ac7d-cd5df12b443c)
2. [Daily Rainfall Dataset](https://ckandev.indiadataportal.com/dataset/climate-data/resource/d97707e6-9309-46a5-ac31-e8482ce8f9eb)
3. [Area, Production and Yield Dataset](https://ckandev.indiadataportal.com/dataset/area-production-yield-apy/resource/f980409d-49a2-42ae-9eb0-182365005c04)

**Study period:** 2017–2021

## Data Processing

The following preprocessing operations were performed:

- Inspected dataset structure, data types and missing values.
- Selected a common five-year study period.
- Removed invalid agricultural records and records with missing production.
- Excluded Total-season records to avoid double-counting.
- Retained valid zero-yield observations.
- Aggregated daily rainfall into agricultural-year features.
- Aggregated groundwater measurements into district-year features.
- Integrated the three datasets using state code, district code and agricultural year.

### Environmental Features

**Rainfall features**
- Total rainfall
- Mean rainfall
- Number of observed rainfall days

**Groundwater features**
- Median groundwater level
- Mean groundwater level
- Mean groundwater level change
- Number of monitoring stations
- Number of groundwater measurements

### Final Integrated Dataset

| Description | Value |
|---|---:|
| Records | 91,809 |
| Columns | 22 |
| Crop categories | 61 |
| Study period | 2017–2021 |

The target variable is `yield`, measured in tonnes per hectare.

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the distribution and characteristics of agricultural yield.

The analysis includes:

- Crop yield distribution
- Log-transformed yield distribution
- Crop-wise yield comparison
- Season-wise yield comparison
- Year-wise yield trends
- Correlation heatmap
- Rainfall versus crop yield
- Groundwater versus crop yield

## Machine Learning Models

Five regression algorithms were implemented and compared:

| No. | Algorithm |
|---:|---|
| 1 | Linear Regression |
| 2 | Decision Tree Regression |
| 3 | Random Forest Regression |
| 4 | Gradient Boosting Regression |
| 5 | Linear Support Vector Regression (Linear SVR) |

### Model Preparation

- Target variable: `yield`
- Numerical features: Agricultural year and environmental indicators
- Categorical features: State code, district code, crop name and season
- Missing numerical values: Median imputation
- Missing categorical values: Most-frequent imputation
- Categorical encoding: One-Hot Encoding
- Numerical scaling: StandardScaler
- Target transformation: Logarithmic transformation using `log1p()`

### Training and Testing Strategy

A chronological split was used:

| Dataset | Years | Records |
|---|---|---:|
| Training | 2017–2020 | 73,472 |
| Testing | 2021 | 18,337 |

This approach evaluates model performance on a later agricultural year rather than randomly mixing observations from different years.

## Model Evaluation

The models are evaluated using the following regression metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute prediction error.
- **MSE (Mean Squared Error):** Measures the average squared prediction error.
- **RMSE (Root Mean Squared Error):** Measures prediction error in the original target unit.
- **R² Score:** Measures the proportion of target variation explained by the model.

Model comparison, actual-versus-predicted plots, residual analysis and feature importance are used to interpret the results.

## Technology Stack

| Technology | Usage |
|---|---|
| Python | Programming language |
| Pandas | Data processing and manipulation |
| NumPy | Numerical computations |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Scikit-learn | Machine learning and preprocessing |
| Spyder | Development environment |

## Project Structure

```text
agricultural-productivity-prediction/
│
├── raw_dataset/
│   ├── apy.csv
│   ├── groundwater.csv
│   └── rainfall_daily.csv
│
├── processed_data/
│   ├── apy_cleaned.csv
│   ├── rainfall_yearly.csv
│   ├── groundwater_yearly.csv
│   └── integrated_agriculture_dataset.csv
│
├── outputs/
│   ├── graphs/
│   └── model_results/
│
├── 01_preprocessing.py
├── 02_exploration.py
├── main.py
│
└── README.md
```

*Note: The raw and processed datasets may be kept locally rather than committed to GitHub because of their file sizes.*

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/agricultural-productivity-prediction.git
```

### 2. Navigate to the project directory

```bash
cd agricultural-productivity-prediction
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Add the datasets

Download the three datasets from the links provided above and place them inside the `raw_dataset/` directory.

### 5. Run preprocessing

```bash
python 01_preprocessing.py
```

### 6. Run exploratory analysis

```bash
python 02_exploration.py
```

### 7. Train and evaluate the models

```bash
python main.py
```

The scripts generate processed datasets, visualisations, model comparison results and prediction outputs.

## Limitations

- Some district-year combinations have missing environmental measurements.
- Changes in district boundaries and geographical identifiers may affect data matching.
- The testing dataset represents only agricultural year 2021.
- Extreme yield observations may influence model evaluation.
- Environmental features are taken from the same agricultural year as the target yield. Therefore, the current implementation performs retrospective yield estimation rather than pre-season forecasting.
- Factors such as soil quality, irrigation, temperature and fertilizer usage are not included.

## Future Scope

- Incorporate additional agricultural and environmental features.
- Explore crop-specific prediction models.
- Evaluate models across multiple future agricultural years.
- Perform hyperparameter tuning to improve model performance.
- Develop an interactive dashboard for district-level agricultural analysis.
- Extend the project towards pre-season crop yield forecasting.

## Author

**Naja Hindh K.**

B.Tech – Computer Science and Engineering

Lovely Professional University

---

*This project was developed as part of the INT234 – Predictive Analytics coursework.*