"""
Training pipeline for crash risk predictor.
Handles temporal split, imbalance, and SHAP explainer creation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import classification_report, roc_auc_score
import shap

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.models.crash_predictor.crash_labeler import CrashLabeler
from scripts.models.crash_predictor.crash_feature_engineer import CrashFeatureEngineer
from scripts.models.crash_predictor.crash_risk_models import PreRaceCrashModel

def main():
    print("🚀 Starting Crash Risk Predictor Training")
    data_dir = project_root / 'data' / 'processed'
    models_dir = project_root / 'models'
    metadata_path = project_root / 'data' / 'cnn' / 'circuit_metadata.csv'
    driver_profiles_path = models_dir / 'driver_style_profiles.csv'
    historical_path = data_dir / 'historical_races_processed.csv'
    status_path = project_root / 'data' / 'raw' / 'kaggle' / 'historical' / 'status.csv'

    print("\n1. Labeling crashes...")
    try:
        labeler = CrashLabeler(historical_path, status_path)
    except FileNotFoundError as e:
        print(e)
        print("Please provide the correct path to status.csv")
        return

    labeled_df = labeler.label_crashes()
    stats = labeler.aggregate_statistics()
    labeler.save_labeled_data(data_dir / 'historical_with_crash_labels.csv')
    labeler.save_statistics(models_dir / 'crash_statistics.json')

    print("\n2. Engineering features...")
    engineer = CrashFeatureEngineer(
        labeled_df,
        circuit_metadata_path=metadata_path,
        driver_profiles_path=driver_profiles_path,
        crash_stats=stats
    )
    feature_df = engineer.engineer_features()
    feature_columns = engineer.feature_columns

    print("\n3. Splitting data temporally...")
    train_df = feature_df[feature_df['year'] <= 2018]
    val_df = feature_df[(feature_df['year'] >= 2019) & (feature_df['year'] <= 2020)]
    test_df = feature_df[feature_df['year'] >= 2021]

    X_train = train_df[feature_columns]
    y_train = train_df['crash_occurred']
    X_val = val_df[feature_columns]
    y_val = val_df['crash_occurred']
    X_test = test_df[feature_columns]
    y_test = test_df['crash_occurred']

    print(f"   Train: {len(X_train)} samples, crashes: {y_train.sum()}")
    print(f"   Val:   {len(X_val)} samples, crashes: {y_val.sum()}")
    print(f"   Test:  {len(X_test)} samples, crashes: {y_test.sum()}")

    neg = (y_train == 0).sum()
    pos = y_train.sum()
    scale_pos_weight = neg / pos if pos > 0 else 1
    print(f"\n   Scale pos weight: {scale_pos_weight:.2f}")

    print("\n4. Training XGBoost classifier...")
    model = PreRaceCrashModel()
    model.train(X_train, y_train, scale_pos_weight=scale_pos_weight)

    print("\n5. Validation evaluation:")
    y_pred_val = (model.model.predict_proba(X_val)[:, 1] > 0.3).astype(int)
    print(classification_report(y_val, y_pred_val, target_names=['No Crash', 'Crash']))
    y_prob_val = model.model.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, y_prob_val)
    print(f"   ROC-AUC: {auc_val:.4f}")

    print("\n6. Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_test)

    print("\n7. Saving artifacts...")
    model.save(models_dir / 'crash_risk_classifier.pkl')
    joblib.dump(feature_columns, models_dir / 'crash_feature_columns.pkl')
    joblib.dump(explainer, models_dir / 'crash_shap_explainer.pkl')
    print("✅ All artifacts saved.")

    print("\n8. Test set evaluation:")
    y_pred_test = (model.model.predict_proba(X_test)[:, 1] > 0.3).astype(int)
    print(classification_report(y_test, y_pred_test, target_names=['No Crash', 'Crash']))
    y_prob_test = model.model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_prob_test)
    print(f"   ROC-AUC: {auc_test:.4f}")

    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()