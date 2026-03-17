"""
train_model.py
CargoCredit – XchangeBox Perishable Trade Finance Risk Engine
Trains two models:
  1. Spoilage classifier  → probability cargo spoils in transit
  2. Default classifier   → probability borrower defaults on invoice
Run once: python train_model.py
"""

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, classification_report,
                              mean_squared_error, r2_score)

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv('cargo_data.csv')
print(f"Loaded {len(df)} shipments")

# ── Features ──────────────────────────────────────────────────────────────
NUM_FEATURES = [
    'distance_km', 'transit_hrs', 'delay_hrs', 'time_ratio',
    'has_cold_chain', 'avg_temp', 'max_temp', 'temp_exceedance',
    'avg_humidity', 'vibration_score', 'cargo_kg', 'prior_defaults',
]
CAT_FEATURES = ['product', 'road_quality', 'borrower_history']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(),                                   NUM_FEATURES),
    ('cat', OneHotEncoder(drop='first', sparse_output=False),   CAT_FEATURES),
])

X = df[NUM_FEATURES + CAT_FEATURES]

X_train, X_test, idx_train, idx_test = train_test_split(
    X, df.index, test_size=0.2, random_state=42
)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p  = preprocessor.transform(X_test)

cat_names    = list(preprocessor.named_transformers_['cat']
                    .get_feature_names_out(CAT_FEATURES))
all_features = NUM_FEATURES + cat_names

# ── Helper ────────────────────────────────────────────────────────────────
def train_and_compare(y_train, y_test, label):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Random Forest':       RandomForestClassifier(
                                   n_estimators=100, max_depth=10,
                                   random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(
                                   n_estimators=150, max_depth=4,
                                   learning_rate=0.1, random_state=42),
    }
    results  = {}
    trained  = {}
    print(f"\n── {label} ──────────────────────────────────")
    for name, m in models.items():
        m.fit(X_train_p, y_train)
        y_pred  = m.predict(X_test_p)
        y_proba = m.predict_proba(X_test_p)[:, 1]
        auc     = roc_auc_score(y_test, y_proba)
        recall  = classification_report(y_test, y_pred,
                                        output_dict=True)['1']['recall']
        acc     = (y_pred == y_test).mean()
        results[name] = {'AUC': round(auc,4),
                         'Recall': round(recall,4),
                         'Accuracy': round(acc,4)}
        trained[name] = m
        print(f"  {name:22}  AUC: {auc:.4f}  Recall: {recall:.4f}")

    best_name  = max(results, key=lambda k: results[k]['AUC'])
    best_model = trained[best_name]
    print(f"  ✓ Best: {best_name}")
    return best_model, best_name, results, trained

# ── Train spoilage model ──────────────────────────────────────────────────
y_spoil_train = df.loc[idx_train, 'spoiled']
y_spoil_test  = df.loc[idx_test,  'spoiled']

spoil_model, spoil_best, spoil_results, spoil_trained = train_and_compare(
    y_spoil_train, y_spoil_test, 'Spoilage Model'
)

# ── Train default model ───────────────────────────────────────────────────
y_def_train = df.loc[idx_train, 'defaulted']
y_def_test  = df.loc[idx_test,  'defaulted']

def_model, def_best, def_results, def_trained = train_and_compare(
    y_def_train, y_def_test, 'Default Model'
)

# ── Feature importances ───────────────────────────────────────────────────
def get_importances(model):
    if hasattr(model, 'feature_importances_'):
        imp = dict(zip(all_features, model.feature_importances_.tolist()))
        return dict(sorted(imp.items(), key=lambda x: -x[1])[:10])
    return {}

# ── Save artefacts ────────────────────────────────────────────────────────
joblib.dump(preprocessor,  'preprocessor.pkl')
joblib.dump(spoil_model,   'spoilage_model.pkl')
joblib.dump(def_model,     'default_model.pkl')
joblib.dump(all_features,  'feature_names.pkl')

meta = {
    'spoilage': {
        'best': spoil_best,
        'comparison': spoil_results,
        'importances': get_importances(spoil_model),
    },
    'default': {
        'best': def_best,
        'comparison': def_results,
        'importances': get_importances(def_model),
    },
    'data_stats': {
        'n_shipments':   len(df),
        'spoilage_rate': round(df['spoiled'].mean(), 4),
        'default_rate':  round(df['defaulted'].mean(), 4),
        'avg_invoice':   round(df['invoice_ngn'].mean(), 0),
        'avg_rate':      round(df['financing_rate'].mean(), 2),
    }
}

with open('model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\n✓ Saved: preprocessor.pkl, spoilage_model.pkl,")
print("         default_model.pkl, feature_names.pkl, model_meta.json")