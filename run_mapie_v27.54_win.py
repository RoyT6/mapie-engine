"""
MAPIE Validation for BFD V27.54 (Windows Compatible)
=====================================================

Validates BFD database against ground truth training data.
Tests anti-cheat bounds and MAPE calculations.

Created: 2026-01-25
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

print("=" * 80)
print("MAPIE VALIDATION - BFD V27.54")
print(f"Timestamp: {datetime.now().isoformat()}")
print("=" * 80)

# Configuration
BASE_DIR = Path(r"C:\Users\RoyT6\Downloads")
BFD_PATH = BASE_DIR / "BFD_V27.54.parquet"
VIEWS_DIR = BASE_DIR / "Views TRaining Data"
COMP_DIR = BASE_DIR / "Components"
MAPIE_DIR = BASE_DIR / "MAPIE"

# Anti-cheat bounds
VALID_MAPE_MIN = 5.0
VALID_MAPE_MAX = 40.0
VALID_R2_MIN = 0.30
VALID_R2_MAX = 0.90


def fmt(seconds: float) -> str:
    """Format seconds as Xm Ys"""
    return f'{int(seconds//60)}m {int(seconds%60)}s'


def parse_views_value(val) -> float:
    """Parse views value, handling comma-formatted numbers."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    # Handle string with commas: "396,900,000" -> 396900000
    try:
        return float(str(val).replace(',', '').strip())
    except (ValueError, TypeError):
        return 0.0


def load_ground_truth() -> Dict[str, float]:
    """Load ground truth from training data sources."""
    print("\n" + "-" * 60)
    print("LOADING GROUND TRUTH")
    print("-" * 60)

    ground_truth = {}
    sources_loaded = 0

    # CSV sources in Views TRaining Data/
    csv_sources = [
        # (relative_path, id_cols, view_cols)
        ('ETL/ETL_trueviews.csv', ['imdb_id', 'id', 'title'], ['views', 'total_views']),
        ('ETL/ETL_trueviews_with_ids.csv', ['imdb_id', 'id'], ['views', 'total_views']),
        ('NETFLIX 2023 2024 wGENRES.csv', ['imdb_id', 'title'], ['views', 'hours_viewed']),
        ('netflix_training_data_per_season.csv', ['imdb_id', 'fc_uid', 'title'], ['views', 'total_views']),
        ('netflix_parsed_all_periods.csv', ['imdb_id', 'fc_uid'], ['views', 'hours_viewed']),
        ('Independent_Non_Netflix_Views_Data.csv', ['imdb_id', 'title'], ['views', 'total_views']),
        ('FlixPatrol/FlixPatrol_Views_v40.10.csv', ['imdb_id', 'fc_uid', 'title'], ['total_views', 'views']),
        ('FlixPatrol/FlixPatrol All Views Jan 25.csv', ['imdb_id', 'fc_uid', 'title'], ['total_views', 'views']),
        ('IMDB Seasons Allocated Correct.csv', ['imdb_id', 'fc_uid'], ['views', 'total_views']),
        ('Canon Customers/essential_top_1004.csv', ['imdb_id', 'title'], ['views', 'total_views']),
    ]

    for rel_path, id_cols, view_cols in csv_sources:
        fpath = VIEWS_DIR / rel_path
        if fpath.exists():
            try:
                df = pd.read_csv(fpath, low_memory=False)

                # Find ID column
                id_col = None
                for col in id_cols:
                    if col in df.columns:
                        id_col = col
                        break
                    for c in df.columns:
                        if col.lower() in c.lower():
                            id_col = c
                            break
                    if id_col:
                        break

                # Find views column
                views_col = None
                for col in view_cols:
                    if col in df.columns:
                        views_col = col
                        break
                    for c in df.columns:
                        if col.lower() in c.lower():
                            views_col = c
                            break
                    if views_col:
                        break

                if id_col and views_col:
                    count = 0
                    for _, row in df.iterrows():
                        key = str(row[id_col]).strip() if pd.notna(row[id_col]) else ''
                        if not key or key == 'nan':
                            continue

                        # Parse views value (handles commas)
                        views = parse_views_value(row[views_col])

                        # Convert hours to views if needed (1 hour ≈ 40,000 views)
                        if 'hour' in views_col.lower() and views < 1000000:
                            views = views * 40000

                        if views > 0:
                            if key in ground_truth:
                                ground_truth[key] = (ground_truth[key] + views) / 2
                            else:
                                ground_truth[key] = views
                            count += 1

                    if count > 0:
                        sources_loaded += 1
                        print(f"  [OK] {rel_path}: {count:,} records")

            except Exception as e:
                print(f"  [SKIP] {rel_path}: {e}")

    # Load parquet sources
    parquet_sources = [
        ('FlixPatrol_Views_Season_Allocated_COMPLETE.parquet', 'fc_uid', 'total_views'),
        ('Independent_Non_Netflix_Views_Data_20260121_171954.parquet', 'imdb_id', 'views'),
        ('FlixPatrol/FlixPatrol_Views_Season_Allocated.parquet', 'fc_uid', 'total_views'),
    ]

    for rel_path, id_col, views_col in parquet_sources:
        fpath = VIEWS_DIR / rel_path
        if fpath.exists():
            try:
                df = pd.read_parquet(fpath)
                count = 0

                # Try to find the columns
                actual_id_col = id_col if id_col in df.columns else None
                actual_views_col = views_col if views_col in df.columns else None

                if not actual_id_col:
                    for c in df.columns:
                        if 'imdb' in c.lower() or 'fc_uid' in c.lower():
                            actual_id_col = c
                            break
                if not actual_views_col:
                    for c in df.columns:
                        if 'view' in c.lower():
                            actual_views_col = c
                            break

                if actual_id_col and actual_views_col:
                    for _, row in df.iterrows():
                        key = str(row[actual_id_col]).strip() if pd.notna(row[actual_id_col]) else ''
                        if not key or key == 'nan':
                            continue

                        views = parse_views_value(row[actual_views_col])

                        if views > 0:
                            if key in ground_truth:
                                ground_truth[key] = (ground_truth[key] + views) / 2
                            else:
                                ground_truth[key] = views
                            count += 1

                    if count > 0:
                        sources_loaded += 1
                        print(f"  [OK] {rel_path}: {count:,} records")

            except Exception as e:
                print(f"  [SKIP] {rel_path}: {e}")

    print(f"\n  Total ground truth: {len(ground_truth):,} records")
    print(f"  Sources loaded: {sources_loaded}")

    return ground_truth


def run_validation(df: pd.DataFrame, ground_truth: Dict[str, float]) -> Dict:
    """Run MAPIE validation."""
    print("\n" + "-" * 60)
    print("RUNNING VALIDATION")
    print("-" * 60)

    results = {
        'timestamp': datetime.now().isoformat(),
        'database': 'BFD_V27.54.parquet',
        'rows': len(df),
        'columns': len(df.columns),
    }

    # Find abstract signal columns
    abs_cols = [c for c in df.columns if c.startswith('abs_')]
    print(f"  Abstract signal columns: {len(abs_cols)}")

    if len(abs_cols) == 0:
        print("  [ERROR] No abstract signal columns found!")
        results['status'] = 'ERROR'
        results['error'] = 'No abstract signal columns'
        return results

    # Match ground truth to database
    print("\n  Matching ground truth...")

    # Try fc_uid first, then imdb_id
    id_cols = ['fc_uid', 'imdb_id']
    matched_count = 0
    y_true = pd.Series(0.0, index=df.index)

    for id_col in id_cols:
        if id_col in df.columns:
            for idx, row in df.iterrows():
                key = str(row[id_col]).strip() if pd.notna(row[id_col]) else ''
                if key in ground_truth and y_true[idx] == 0:
                    y_true[idx] = ground_truth[key]
                    matched_count += 1

    print(f"  Ground truth matched: {matched_count:,} records")
    results['ground_truth_matched'] = matched_count

    if matched_count < 1000:
        print("  [WARN] Low ground truth matches, using views_computed as fallback")
        if 'views_computed' in df.columns:
            mask = (y_true == 0) & (df['views_computed'] > 0)
            y_true[mask] = df.loc[mask, 'views_computed']
            matched_count = (y_true > 0).sum()
            print(f"  Updated matches with views_computed: {matched_count:,}")

    # Apply temporal filter
    print("\n  Applying temporal filter...")
    current_year = datetime.now().year
    mask = y_true > 0

    if 'status' in df.columns:
        unreleased = ['Upcoming', 'Announced', 'In Production', 'Post Production']
        mask = mask & ~df['status'].isin(unreleased)

    if 'start_year' in df.columns:
        mask = mask & (df['start_year'].fillna(0) <= current_year)

    filtered_count = mask.sum()
    print(f"  Rows after filter: {filtered_count:,}")

    if filtered_count < 1000:
        print("  [ERROR] Insufficient filtered records!")
        results['status'] = 'INSUFFICIENT_DATA'
        return results

    # Prepare data for model
    X = df.loc[mask, abs_cols].fillna(0)
    y = y_true[mask]

    # Train/test split
    np.random.seed(42)
    test_size = int(len(X) * 0.2)
    test_idx = np.random.choice(len(X), size=test_size, replace=False)
    train_idx = np.array([i for i in range(len(X)) if i not in test_idx])

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    # Train XGBoost model
    print("\n  Training XGBoost model...")
    t0 = time.time()

    try:
        import xgboost as xgb

        # Check for GPU
        try:
            model = xgb.XGBRegressor(
                tree_method='hist',
                device='cuda',
                n_estimators=1350,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
                verbosity=0
            )
            model.fit(X_train, y_train)
            gpu_used = True
        except Exception:
            model = xgb.XGBRegressor(
                tree_method='hist',
                n_estimators=1350,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
                verbosity=0
            )
            model.fit(X_train, y_train)
            gpu_used = False

        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s (GPU: {gpu_used})")

        # Predict
        pred = model.predict(X_test)

        # Calculate metrics
        non_zero = y_test > 0
        if non_zero.sum() > 100:
            ape = np.abs((y_test[non_zero] - pred[non_zero]) / y_test[non_zero])
            mape = float(np.mean(ape) * 100)
        else:
            ape = np.abs((y_test - pred) / np.maximum(y_test, 1))
            mape = float(np.mean(ape) * 100)

        ss_res = np.sum((y_test - pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0

        # Anti-cheat validation
        r2_valid = VALID_R2_MIN <= r2 <= VALID_R2_MAX
        mape_valid = VALID_MAPE_MIN <= mape <= VALID_MAPE_MAX

        results['metrics'] = {
            'r2': round(r2, 4),
            'mape': round(mape, 2),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_time_seconds': round(train_time, 1),
            'gpu_used': gpu_used
        }

        results['anti_cheat'] = {
            'r2_valid': r2_valid,
            'r2_range': f"[{VALID_R2_MIN}, {VALID_R2_MAX}]",
            'mape_valid': mape_valid,
            'mape_range': f"[{VALID_MAPE_MIN}, {VALID_MAPE_MAX}]",
            'status': 'PASS' if (r2_valid and mape_valid) else 'FAIL'
        }

        results['status'] = 'SUCCESS'

    except ImportError:
        print("  [ERROR] XGBoost not installed!")
        results['status'] = 'ERROR'
        results['error'] = 'XGBoost not installed'
        return results

    except Exception as e:
        print(f"  [ERROR] Model training failed: {e}")
        results['status'] = 'ERROR'
        results['error'] = str(e)
        return results

    return results


def main():
    start_time = time.time()

    # Load BFD
    print("\n" + "-" * 60)
    print("LOADING DATABASE")
    print("-" * 60)

    if not BFD_PATH.exists():
        print(f"[ERROR] BFD not found: {BFD_PATH}")
        sys.exit(1)

    print(f"  File: {BFD_PATH.name}")
    print(f"  Size: {BFD_PATH.stat().st_size / (1024**3):.2f} GB")

    t0 = time.time()
    df = pd.read_parquet(BFD_PATH)
    load_time = time.time() - t0

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Load time: {load_time:.1f}s")

    # Load ground truth
    ground_truth = load_ground_truth()

    # Run validation
    results = run_validation(df, ground_truth)

    # Print summary
    print("\n" + "=" * 80)
    print("MAPIE VALIDATION SUMMARY - BFD V27.54")
    print("=" * 80)

    if results.get('status') == 'SUCCESS':
        metrics = results.get('metrics', {})
        anti_cheat = results.get('anti_cheat', {})

        print(f"\n  R-squared: {metrics.get('r2', 0):.4f} {'PASS' if anti_cheat.get('r2_valid') else 'FAIL'}")
        print(f"  MAPE: {metrics.get('mape', 0):.2f}% {'PASS' if anti_cheat.get('mape_valid') else 'FAIL'}")
        print(f"  Anti-Cheat Status: {anti_cheat.get('status', 'UNKNOWN')}")
        print(f"\n  Train samples: {metrics.get('train_samples', 0):,}")
        print(f"  Test samples: {metrics.get('test_samples', 0):,}")
        print(f"  GPU used: {metrics.get('gpu_used', False)}")

        # V27 compliance check
        v27_compliant = True
        temporal_cols = [c for c in df.columns if c.startswith('views_h') and '_20' in c]
        if len(temporal_cols) < 100:
            v27_compliant = False

        print(f"\n  V27.00 Temporal Compliant: {'YES' if v27_compliant else 'NO'}")
        results['v27_compliant'] = v27_compliant
    else:
        print(f"\n  Status: {results.get('status', 'UNKNOWN')}")
        if 'error' in results:
            print(f"  Error: {results['error']}")

    total_time = time.time() - start_time
    print(f"\n  Total runtime: {fmt(total_time)}")
    print("=" * 80)

    # Save results
    output_file = MAPIE_DIR / f"VALIDATION_V27.54_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {output_file}")

    return results


if __name__ == '__main__':
    main()
