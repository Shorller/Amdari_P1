# Data Handling
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Model Evaluation
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# Display Settings
plt.style.use('seaborn-v0_8')  # Clean plot style

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print ("✅ All libraries imported successfully!")

# Load dataset
data = pd.read_excel('Glucose_Spike_Dataset.xlsx')
print ("✅ Dataset loaded successfully!")

# Shape of the Dataset
print("DATASET SHAPE")
print("=" * 60)
print(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")

print("\nFIRST 5 ROWS")
print(data.head())

print("\nLAST 5 ROWS")
print(data.tail())

# Data Types
print("DATA TYPES")
print("=" * 50)
print(data.dtypes)

# ============================================================
# SECTION 1: MISSING VALUES ANALYSIS
# ============================================================

print("=" * 60)
print("MISSING VALUES ANALYSIS")
print("=" * 60)

missing_count = data.isnull().sum()
missing_percent = (data.isnull().sum() / len(data)) * 100

missing_summary = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing (%)': missing_percent.round(2)
})

missing_summary = missing_summary[missing_summary['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

if missing_summary.empty:
    print("✅ No missing values found in the dataset.")
else:
    print(missing_summary)
    print(f"\nTotal columns with missing values: {len(missing_summary)}")

    # ============================================================
# SECTION 2: DUPLICATE RECORDS CHECK
# ============================================================

print("=" * 60)
print("DUPLICATE RECORDS CHECK")
print("=" * 60)

total_duplicates = data.duplicated().sum()
patient_meal_dupes = data.duplicated(subset=['patient_id', 'meal_time']).sum()

print(f"Total fully duplicate rows     : {total_duplicates}")
print(f"Duplicate patient-meal records : {patient_meal_dupes}")

if total_duplicates > 0:
    print("\n⚠️  Duplicate rows detected. Preview:")
    print(data[data.duplicated()].head())
else:
    print("✅ No fully duplicate rows found.")

if patient_meal_dupes > 0:
    print(f"\n⚠️  {patient_meal_dupes} rows share the same patient_id + meal_time combination.")
else:
    print("✅ No duplicate patient-meal combinations found.")