import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
plt.style.use('seaborn-v0_8-darkgrid')

# Load the data
data = pd.read_csv('emm.csv')

print("=" * 80)
print("MACHINE LEARNING PREPROCESSING - EARLY NEONATAL MORTALITY (ENM)")
print("=" * 80)

# 1. INITIAL DATA EXPLORATION
print("\n1. INITIAL DATA EXPLORATION")
print("-" * 40)
print(f"Dataset shape: {data.shape}")
print(f"\nMissing values check:\n{data.isnull().sum()}")

# ---- EARLY NEONATAL MORTALITY RATE (before any row is dropped) ----
target = 'enm'
enm_counts = data[target].value_counts()
n_total = len(data)
n_deaths = (data[target] == 'Early Neonatal Death').sum()
enm_rate_pct = n_deaths / n_total * 100
enm_rate_per1000 = n_deaths / n_total * 1000

print("\n" + "=" * 80)
print("EARLY NEONATAL MORTALITY (ENM) SUMMARY")
print("=" * 80)
print(enm_counts)
print(f"\nTotal births in dataset: {n_total}")
print(f"Early neonatal deaths:   {n_deaths}")
print(f"ENM rate: {enm_rate_pct:.2f}%")
print(f"ENM rate: {enm_rate_per1000:.1f} per 1,000 live births")

# 2. DATA CLEANING
print("\n" + "=" * 80)
print("2. DATA CLEANING")
print("-" * 40)

duplicates = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
print("NOTE: Not dropping duplicates -- with only categorical variables, repeated")
print("combinations of characteristics are expected across thousands of real")
print("individuals and do not necessarily indicate data errors.")

# husbandedu has a small number of '8' codes (DHS 'don't know' code) -> treat as missing
if 'husbandedu' in data.columns:
    n_unknown = (data['husbandedu'] == '8').sum()
    print(f"\n'husbandedu' contains {n_unknown} rows coded '8' (don't know) -> will be dropped after mapping")

# Constant columns check
constant_columns = [c for c in data.columns if data[c].nunique() == 1]
if constant_columns:
    data = data.drop(columns=constant_columns)
    print(f"Dropped constant columns: {constant_columns}")
else:
    print("No constant columns found")

# 3. SEPARATE TARGET
print("\n" + "=" * 80)
print("3. TARGET / FEATURE SPLIT")
print("-" * 40)

y = data[target]
X = data.drop(columns=[target])

label_encoder_y = LabelEncoder()
# Force consistent ordering so 1 = Early Neonatal Death
label_encoder_y.classes_ = np.array(['No Early Neonatal Death', 'Early Neonatal Death'])
y_encoded = label_encoder_y.transform(y)
print(f"Target encoding: {dict(zip(label_encoder_y.classes_, label_encoder_y.transform(label_encoder_y.classes_)))}")

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")

# 4. FEATURE ENGINEERING
print("\n" + "=" * 80)
print("4. FEATURE ENGINEERING")
print("-" * 40)

X_fe = X.copy()

# Ordinal variables
ordinal_mapping = {
    'magegrp': {'<20 years': 0, '20-34 years': 1, '35+ years': 2},
    'medu': {'No education': 0, 'Primary': 1, 'Secondary': 2, 'Higher': 3},
    'husbandedu': {'No education': 0, 'Primary': 1, 'Secondary': 2, 'Higher': 3, '8': np.nan},
    'wealth': {'Poorest': 0, 'Poorer': 1, 'Middle': 2, 'Richer': 3, 'Richest': 4},
    'media_exposure_cat': {'No media exposure': 0, 'Low-moderate exposure': 1, 'High exposure': 2},
}

for col, mapping in ordinal_mapping.items():
    if col in X_fe.columns:
        X_fe[col] = X_fe[col].map(mapping)
        print(f"Ordinal encoded: {col}")

# Binary variables
binary_mapping = {
    'residence': {'Rural': 0, 'Urban': 1},
    'work': {'No': 0, 'Yes': 1},
    'sex': {'Female': 0, 'Male': 1},
    'facility': {'Home': 0, 'Health facility': 1},
    'cs': {'No': 0, 'Yes': 1},
    'lbw': {'Normal birth weight': 0, 'Low birth weight': 1},
    'pnc': {'No PNC': 0, 'Received PNC': 1},
    'unimproved_water': {'NO': 0, 'yes': 1},
    'unimproved_toilet': {'NO': 0, 'yes': 1},
    'solid_fuel': {'NO': 0, 'yes': 1},
    'mat_underweight': {'NO': 0, 'yes': 1},
    'teen_mother': {'NO': 0, 'yes': 1},
    'short_interval': {'no': 0, 'yes': 1},
    'ANC_Inaducate': {'No': 0, 'yes': 1},
    'birtorder': {'No': 0, 'yes': 1},
}

for col, mapping in binary_mapping.items():
    if col in X_fe.columns:
        X_fe[col] = X_fe[col].map(mapping)
        print(f"Binary encoded: {col}")

# Drop rows with unmapped/missing values introduced above (e.g. husbandedu == '8')
n_before = len(X_fe)
mask = X_fe.notnull().all(axis=1)
X_fe = X_fe[mask]
y_encoded = y_encoded[mask.values]
print(f"\nDropped {n_before - len(X_fe)} rows with unmapped/missing category codes")

remaining_categorical = X_fe.select_dtypes(include=['object']).columns.tolist()
if remaining_categorical:
    X_fe = pd.get_dummies(X_fe, columns=remaining_categorical, drop_first=True)
    print(f"One-hot encoded remaining variables: {remaining_categorical}")

X_fe = X_fe.astype(float)
print(f"\nFinal features shape after encoding: {X_fe.shape}")

# 5. DATA SPLITTING
print("\n" + "=" * 80)
print("5. DATA SPLITTING")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X_fe, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
print(f"Training target distribution: {np.bincount(y_train)}")
print(f"Testing target distribution: {np.bincount(y_test)}")
print("\nNOTE: The outcome is highly imbalanced (~3.5% deaths). Accuracy alone will")
print("look high by default; ROC-AUC and recall on the minority class matter more.")

# 6. FEATURE SCALING
print("\n" + "=" * 80)
print("6. FEATURE SCALING")
print("-" * 40)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_fe.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_fe.columns)
print("Feature scaling completed using StandardScaler")

# 7. FEATURE IMPORTANCE
print("\n" + "=" * 80)
print("7. FEATURE IMPORTANCE (Random Forest)")
print("-" * 40)

rf_for_feature_importance = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight='balanced'
)
rf_for_feature_importance.fit(X_train_scaled, y_train)

feature_importance = pd.DataFrame({
    'feature': X_fe.columns,
    'importance': rf_for_feature_importance.feature_importances_
}).sort_values('importance', ascending=False)

print("\nAll features ranked by importance:")
print(feature_importance.to_string(index=False))

plt.figure(figsize=(10, 8))
plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Predicting Early Neonatal Death', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()

# 8. SAVE PREPROCESSED DATA
print("\n" + "=" * 80)
print("8. SAVING PREPROCESSED DATA")
print("-" * 40)

processed_data = X_fe.copy()
processed_data[target] = y_encoded
processed_data.to_csv('preprocessed_emm.csv', index=False)

train_data = X_train_scaled_df.copy()
train_data[target] = y_train
test_data = X_test_scaled_df.copy()
test_data[target] = y_test
train_data.to_csv('train_data_scaled.csv', index=False)
test_data.to_csv('test_data_scaled.csv', index=False)
print("Saved: preprocessed_emm.csv, train_data_scaled.csv, test_data_scaled.csv")

# 9. BASELINE MODELS (class_weight balanced due to rare outcome)
print("\n" + "=" * 80)
print("9. BASELINE MODEL EVALUATION")
print("-" * 40)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    results[name] = {
        'model': model, 'accuracy': accuracy, 'roc_auc': roc_auc,
        'predictions': y_pred, 'probabilities': y_pred_proba
    }
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC AUC:  {roc_auc:.4f}")
    print("  Classification Report:")
    print("  " + classification_report(
        y_test, y_pred, target_names=label_encoder_y.classes_
    ).replace('\n', '\n  '))

# 10. MODEL COMPARISON PLOT
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
model_names = list(results.keys())
accuracies = [results[n]['accuracy'] for n in model_names]

bars = axes[0].bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
axes[0].set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.4f}', ha='center', fontweight='bold')

for name in model_names:
    fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.4f})", linewidth=2)
axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1].set_title('ROC Curves', fontsize=13, fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 11. SUMMARY
best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
best_auc = results[best_model_name]['roc_auc']
best_acc = results[best_model_name]['accuracy']

summary = f"""
EARLY NEONATAL MORTALITY (ENM) - PREPROCESSING & MODELING SUMMARY
====================================================================
Original dataset: {n_total} births
Early neonatal deaths: {n_deaths}
ENM rate: {enm_rate_pct:.2f}% ({enm_rate_per1000:.1f} per 1,000 live births)

Processed features shape: {X_fe.shape}
Training set: {X_train.shape}, Testing set: {X_test.shape}

Baseline models evaluated: {', '.join(model_names)}
Best model (by ROC-AUC): {best_model_name} -> AUC = {best_auc:.4f}, Accuracy = {best_acc:.4f}

Top 5 predictive features:
{feature_importance.head(5).to_string(index=False)}
"""
print(summary)

with open('preprocessing_summary.txt', 'w') as f:
    f.write(summary)

print("\nDONE.")
