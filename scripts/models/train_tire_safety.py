"""
Train the tire safety risk predictor.
"""

import sys
from pathlib import Path
import pandas as pd
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.models.tire_safety import TireSafetyPredictor

def main():
    print("🚀 Starting Tire Safety Model Training")
    # Load modern lap data
    data_path = project_root / 'data' / 'processed' / 'modern_with_historical_context.csv'
    df = pd.read_csv(data_path)
    print(f"📥 Loaded {len(df)} laps.")

    # Initialize predictor
    predictor = TireSafetyPredictor()

    # Engineer features and risk labels
    print("🔄 Engineering features...")
    df_eng = predictor.engineer_features(df)
    print(f"✅ Features engineered. Risk score range: {df_eng['risk_score'].min():.1f}-{df_eng['risk_score'].max():.1f}")

    # Prepare training data
    X, y = predictor.prepare_training_data(df_eng)
    print(f"📊 Training samples: {X.shape[0]}, features: {X.shape[1]}")

    # Train model
    predictor.train(X, y)

    # Save model
    predictor.save_model()
    print("🎉 Tire safety model training complete!")

if __name__ == "__main__":
    main()