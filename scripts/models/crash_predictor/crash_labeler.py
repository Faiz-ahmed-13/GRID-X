"""
Crash Labeler – Extracts crash events from historical race data.
Uses statusId to join with status.csv for crash keywords.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

class CrashLabeler:
    CRASH_KEYWORDS = [
        'accident', 'collision', 'crash', 'spun', 'damage', 'contact',
        'barrier', 'retired', 'wreck', 'hit', 'stalled', 'debris',
        'tyre failure', 'suspension', 'broken wing', 'off track'
    ]

    def __init__(self, historical_csv_path, status_csv_path=None):
        self.df = pd.read_csv(historical_csv_path, low_memory=False)
        self.df['crash_occurred'] = 0

        if status_csv_path is None:
            hist_path = Path(historical_csv_path)
            possible = [
                hist_path.parent.parent / 'raw' / 'kaggle' / 'historical' / 'status.csv',
                hist_path.parent / 'status.csv',
                hist_path.parent.parent / 'status.csv'
            ]
            for p in possible:
                if p.exists():
                    status_csv_path = p
                    break
            if status_csv_path is None:
                raise FileNotFoundError(
                    "status.csv not found. Please provide its path as an argument.\n"
                    "Available columns in historical data:\n" + str(self.df.columns.tolist())
                )

        self.statuses = pd.read_csv(status_csv_path)
        self.df = self.df.merge(self.statuses[['statusId', 'status']], on='statusId', how='left')

    def label_crashes(self):
        if 'status' not in self.df.columns:
            raise ValueError("Status column not found after merge.")
        statuses = self.df['status'].astype(str).str.lower()
        crash_mask = statuses.str.contains('|'.join(self.CRASH_KEYWORDS), na=False)
        self.df.loc[crash_mask, 'crash_occurred'] = 1
        return self.df

    def aggregate_statistics(self):
        stats = {}
        total_races = len(self.df)
        total_crashes = self.df['crash_occurred'].sum()
        stats['crash_rate'] = total_crashes / total_races if total_races > 0 else 0
        stats['total_crashes'] = int(total_crashes)

        if 'circuitId' in self.df.columns:
            circuit_stats = self.df.groupby('circuitId')['crash_occurred'].agg(['mean', 'count']).to_dict('index')
            stats['circuit_crash_rates'] = {int(k): v['mean'] for k, v in circuit_stats.items()}
            stats['circuit_race_counts'] = {int(k): int(v['count']) for k, v in circuit_stats.items()}
        else:
            stats['circuit_crash_rates'] = {}
            stats['circuit_race_counts'] = {}

        if 'driverRef' in self.df.columns:
            driver_stats = self.df.groupby('driverRef')['crash_occurred'].agg(['mean', 'count']).to_dict('index')
            stats['driver_crash_rates'] = {k: v['mean'] for k, v in driver_stats.items()}
            stats['driver_race_counts'] = {k: int(v['count']) for k, v in driver_stats.items()}
        else:
            stats['driver_crash_rates'] = {}
            stats['driver_race_counts'] = {}

        if 'weather' in self.df.columns:
            wet_mask = self.df['weather'].str.contains('rain|wet', na=False)
            wet_crashes = self.df[wet_mask]['crash_occurred'].sum()
            wet_races = wet_mask.sum()
            stats['wet_crash_rate'] = wet_crashes / wet_races if wet_races > 0 else 0
            dry_crashes = self.df[~wet_mask]['crash_occurred'].sum()
            dry_races = (~wet_mask).sum()
            stats['dry_crash_rate'] = dry_crashes / dry_races if dry_races > 0 else 0
        else:
            stats['wet_crash_rate'] = 0
            stats['dry_crash_rate'] = 0

        return stats

    def save_labeled_data(self, output_path):
        self.df.to_csv(output_path, index=False)
        print(f"✅ Labeled data saved to {output_path}")

    def save_statistics(self, output_path):
        stats = self.aggregate_statistics()
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Statistics saved to {output_path}")