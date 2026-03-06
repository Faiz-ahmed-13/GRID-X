"""
Crash Feature Engineer – Transforms raw data into features for crash prediction.
Handles circuit metadata with mixed delimiter (space header, comma data).
"""

import pandas as pd
import numpy as np
from pathlib import Path

def read_circuit_metadata(filepath):
    """
    Read circuit metadata CSV with space-separated header and comma-separated data.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        col_names = first_line.split()  # space-separated headers
        # Read the rest with pandas, skip first row
        df = pd.read_csv(filepath, skiprows=1, header=None, names=col_names, encoding='utf-8')
    df.columns = df.columns.str.strip()
    return df

class CrashFeatureEngineer:
    def __init__(self, labeled_df, circuit_metadata_path, driver_profiles_path, crash_stats):
        self.df = labeled_df.copy()
        # Read circuit metadata correctly
        self.circuit_metadata = read_circuit_metadata(circuit_metadata_path)
        self.driver_profiles = pd.read_csv(driver_profiles_path)
        self.crash_stats = crash_stats
        self.feature_columns = None

        print("📋 Circuit metadata columns:", self.circuit_metadata.columns.tolist())

    def _add_circuit_features(self):
        # Historical data has circuitRef (e.g., 'albert_park') which matches circuit_name in metadata
        # Create mapping from circuitRef to metadata fields
        circuit_map = self.circuit_metadata.set_index('circuit_name')[['track_type', 'corners', 'length_km']].to_dict('index')

        # Apply mapping
        self.df['track_type'] = self.df['circuitRef'].map({k: v['track_type'] for k, v in circuit_map.items()}).fillna('permanent')
        self.df['corners'] = self.df['circuitRef'].map({k: v['corners'] for k, v in circuit_map.items()}).fillna(15)
        self.df['length_km'] = self.df['circuitRef'].map({k: v['length_km'] for k, v in circuit_map.items()}).fillna(5.0)

        # Encode track type
        track_type_map = {'street': 2, 'permanent': 1, 'road': 0, 'Semi-Permanent Circuit': 1, 'Temporary Circuit': 1}
        self.df['circuit_type_encoded'] = self.df['track_type'].map(track_type_map).fillna(1)

        # Corner density
        self.df['corner_density'] = self.df['corners'] / self.df['length_km']

        # Circuit historical crash rate (from crash_stats) – we need circuitId for that
        # Use circuitId mapping if available
        if 'circuitId' in self.df.columns:
            self.df['circuit_historical_crash_rate'] = self.df['circuitId'].map(
                self.crash_stats['circuit_crash_rates']
            ).fillna(self.crash_stats['crash_rate'])
        else:
            self.df['circuit_historical_crash_rate'] = self.crash_stats['crash_rate']

    def _add_driver_features(self):
        self.df = self.df.merge(
            self.driver_profiles[['Driver', 'AggressionScore', 'style_label']],
            left_on='driverRef', right_on='Driver', how='left'
        )
        self.df['driver_historical_crash_rate'] = self.df['driverRef'].map(
            self.crash_stats['driver_crash_rates']
        ).fillna(self.crash_stats['crash_rate'])

        # Driver experience (years since first race)
        self.df['first_year'] = self.df.groupby('driverRef')['year'].transform('min')
        self.df['driver_experience'] = self.df['year'] - self.df['first_year']
        self.df['driver_experience'] = self.df['driver_experience'].fillna(0)

    def _add_race_context_features(self):
        self.df['grid'] = pd.to_numeric(self.df['grid'], errors='coerce').fillna(20)

        if 'points' in self.df.columns:
            leader_points = self.df.groupby(['year', 'round'])['points'].transform('max')
            self.df['points_gap_to_leader'] = leader_points - self.df['points']
            self.df['points_gap_to_leader'] = self.df['points_gap_to_leader'].fillna(0)
        else:
            self.df['points_gap_to_leader'] = 0

    def _add_weather_features(self):
        if 'weather' in self.df.columns:
            wet_keywords = ['rain', 'wet', 'shower']
            self.df['weather_wet'] = self.df['weather'].str.lower().str.contains(
                '|'.join(wet_keywords), na=False
            ).astype(int)
        else:
            self.df['weather_wet'] = 0
        self.df['track_temp'] = 0  # placeholder

    def engineer_features(self):
        self._add_circuit_features()
        self._add_driver_features()
        self._add_race_context_features()
        self._add_weather_features()

        self.feature_columns = [
            'circuit_type_encoded',
            'circuit_historical_crash_rate',
            'corner_density',
            'AggressionScore',
            'driver_historical_crash_rate',
            'driver_experience',
            'grid',
            'points_gap_to_leader',
            'weather_wet',
            'track_temp'
        ]

        for col in self.feature_columns:
            if col not in self.df.columns:
                self.df[col] = 0
            self.df[col] = self.df[col].fillna(self.df[col].median())

        # Return features + crash_occurred + year (for splitting)
        return self.df[self.feature_columns + ['crash_occurred', 'year']]