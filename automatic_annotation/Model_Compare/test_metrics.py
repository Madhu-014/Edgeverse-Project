"""
Test script to generate sample metrics data
"""
import pandas as pd
from datetime import datetime
import os

# Sample metrics with per-class data
sample_metrics = [
    {
        "run_id": "20260218_120000",
        "model_name": "v1",
        "date": "2026-02-18 12:00:00",
        "overall_precision": 0.85,
        "overall_recall": 0.82,
        "precision_person": 0.90,
        "recall_person": 0.88,
        "precision_bicycle": 0.82,
        "recall_bicycle": 0.79,
        "precision_car": 0.88,
        "recall_car": 0.85,
        "precision_motorbike": 0.75,
        "recall_motorbike": 0.72,
        "precision_bus": 0.93,
        "recall_bus": 0.91,
        "precision_truck": 0.87,
        "recall_truck": 0.84,
        "precision_auto": 0.81,
        "recall_auto": 0.78,
    },
    {
        "run_id": "20260218_120100",
        "model_name": "v2",
        "date": "2026-02-18 12:01:00",
        "overall_precision": 0.88,
        "overall_recall": 0.86,
        "precision_person": 0.92,
        "recall_person": 0.90,
        "precision_bicycle": 0.85,
        "recall_bicycle": 0.83,
        "precision_car": 0.91,
        "recall_car": 0.89,
        "precision_motorbike": 0.79,
        "recall_motorbike": 0.76,
        "precision_bus": 0.95,
        "recall_bus": 0.93,
        "precision_truck": 0.90,
        "recall_truck": 0.87,
        "precision_auto": 0.84,
        "recall_auto": 0.82,
    }
]

# Create DataFrame
df = pd.DataFrame(sample_metrics)

# Save to CSV
csv_path = "metrics.csv"
df.to_csv(csv_path, index=False)

print(f"Sample metrics saved to {csv_path}")
print("\nMetrics preview:")
print(df.to_string())
