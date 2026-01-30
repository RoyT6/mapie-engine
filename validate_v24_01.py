#!/usr/bin/env python3
"""
MAPIE VALIDATION TEST FOR BFD_V24.01
====================================
Runs validation tests against BFD_V24.01.parquet
"""
import os
os.environ['CUDF_SPILL'] = 'on'
os.environ['PYTHONUNBUFFERED'] = '1'

import cudf
import cupy as cp
import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy import stats
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

cp.cuda.Device(0).use()
cp.get_default_memory_pool().free_all_blocks()

print('='*80)
print('MAPIE VALIDATION TEST - BFD V24.01')
print('='*80)
print(f'Timestamp: {datetime.now().isoformat()}')
print(f'GPU: {cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}')
print()

BASE = '/mnt/c/Users/RoyT6/Downloads'
BFD_PATH = f'{BASE}/BFD_V24.01.parquet'

results = {'timestamp': datetime.now().isoformat(), 'version': '24.01', 'tests': {}}

# ============================================================================
# SECTION 1: LOAD DATABASE
# ============================================================================
print('[1] LOADING DATABASE...')
t0 = datetime.now()

try:
    bfd = cudf.read_parquet(BFD_PATH)
    load_time = (datetime.now() - t0).total_seconds()
    print(f'  File: BFD_V24.01.parquet')
    print(f'  Rows: {len(bfd):,}')
    print(f'  Columns: {len(bfd.columns)}')
    print(f'  Load time: {load_time:.2f}s')

    # Convert to pandas for analysis
    bfd_pd = bfd.to_pandas()
    del bfd
    cp.get_default_memory_pool().free_all_blocks()

    results['tests']['load'] = {
        'status': 'PASS',
        'rows': len(bfd_pd),
        'columns': len(bfd_pd.columns),
        'load_time_seconds': load_time
    }
except Exception as e:
    print(f'  ERROR: {e}')
    results['tests']['load'] = {'status': 'FAIL', 'error': str(e)}
    exit(1)

# ============================================================================
# SECTION 2: SCHEMA VALIDATION
# ============================================================================
print()
print('='*80)
print('[2] SCHEMA VALIDATION')
print('='*80)

required_cols = ['views_computed', 'title', 'fc_uid']
optional_cols = ['imdb_id', 'tmdb_id', 'genres', 'title_type', 'start_year',
                 'end_year', 'status', 'premiere_date']

found_required = [c for c in required_cols if c in bfd_pd.columns]
found_optional = [c for c in optional_cols if c in bfd_pd.columns]
abs_cols = [c for c in bfd_pd.columns if c.startswith('abs_')]

print(f'  Required columns found: {len(found_required)}/{len(required_cols)}')
for c in required_cols:
    status = 'OK' if c in bfd_pd.columns else 'MISSING'
    print(f'    - {c}: {status}')

print(f'  Optional columns found: {len(found_optional)}/{len(optional_cols)}')
print(f'  Abstract signal columns: {len(abs_cols)}')

results['tests']['schema'] = {
    'status': 'PASS' if len(found_required) == len(required_cols) else 'FAIL',
    'required_found': len(found_required),
    'required_total': len(required_cols),
    'optional_found': len(found_optional),
    'abstract_signals': len(abs_cols)
}

# ============================================================================
# SECTION 3: DATA QUALITY
# ============================================================================
print()
print('='*80)
print('[3] DATA QUALITY')
print('='*80)

if 'views_computed' in bfd_pd.columns:
    views = bfd_pd['views_computed'].values

    # Basic stats
    views_mean = np.mean(views)
    views_std = np.std(views)
    views_min = np.min(views)
    views_max = np.max(views)
    views_median = np.median(views)

    # Null check
    null_count = bfd_pd['views_computed'].isna().sum()
    zero_count = (bfd_pd['views_computed'] == 0).sum()
    negative_count = (bfd_pd['views_computed'] < 0).sum()

    print(f'  views_computed Statistics:')
    print(f'    Mean: {views_mean:,.0f}')
    print(f'    Std Dev: {views_std:,.0f}')
    print(f'    Min: {views_min:,.0f}')
    print(f'    Max: {views_max:,.0f}')
    print(f'    Median: {views_median:,.0f}')
    print(f'    CV: {views_std/views_mean*100:.1f}%')
    print()
    print(f'  Data Issues:')
    print(f'    Nulls: {null_count:,} ({null_count/len(bfd_pd)*100:.2f}%)')
    print(f'    Zeros: {zero_count:,} ({zero_count/len(bfd_pd)*100:.2f}%)')
    print(f'    Negative: {negative_count:,} ({negative_count/len(bfd_pd)*100:.2f}%)')

    results['tests']['data_quality'] = {
        'status': 'PASS' if negative_count == 0 else 'WARNING',
        'views_mean': float(views_mean),
        'views_std': float(views_std),
        'views_min': float(views_min),
        'views_max': float(views_max),
        'null_count': int(null_count),
        'zero_count': int(zero_count),
        'negative_count': int(negative_count)
    }
else:
    print('  ERROR: views_computed column not found')
    results['tests']['data_quality'] = {'status': 'FAIL', 'error': 'views_computed not found'}
    views = np.array([])

# ============================================================================
# SECTION 4: SINE WAVE DETECTION (FRAUD CHECK)
# ============================================================================
print()
print('='*80)
print('[4] SINE WAVE DETECTION (Fraud Check)')
print('='*80)

if len(views) > 0:
    n = len(views)
    sorted_views = np.sort(views)
    residuals = sorted_views - np.linspace(sorted_views.min(), sorted_views.max(), n)

    fft_result = fft(residuals)
    frequencies = fftfreq(n, 1)
    power_spectrum = np.abs(fft_result)**2

    dominant_freq_idx = np.argsort(power_spectrum[1:n//2])[-5:] + 1
    dominant_freqs = frequencies[dominant_freq_idx]
    dominant_powers = power_spectrum[dominant_freq_idx]
    total_power = np.sum(power_spectrum[1:n//2])
    dominant_power_pct = np.sum(dominant_powers) / total_power * 100

    print(f'  FFT Analysis on {n:,} values')
    print(f'  Total dominant frequency power: {dominant_power_pct:.2f}%')
    print()
    print('  Top 5 Frequencies:')
    for i, (freq, power) in enumerate(zip(dominant_freqs[::-1], dominant_powers[::-1])):
        period = int(1/abs(freq)) if freq != 0 else n
        pct = power/total_power*100
        print(f'    {i+1}. Period={period:,} samples, Power={pct:.2f}%')

    max_single_freq = max(dominant_powers)/total_power*100
    sine_detected = max_single_freq > 50

    print()
    print(f'  Max single frequency: {max_single_freq:.2f}%')
    print(f'  Sine Wave Status: {"DETECTED - INVESTIGATE" if sine_detected else "PASS - Natural distribution"}')

    results['tests']['sine_wave'] = {
        'status': 'WARNING' if sine_detected else 'PASS',
        'max_single_freq_pct': float(max_single_freq),
        'interpretation': 'Natural variation' if not sine_detected else 'Potential artificial pattern'
    }
else:
    results['tests']['sine_wave'] = {'status': 'SKIP', 'reason': 'No views data'}

# ============================================================================
# SECTION 5: ABERRATION TESTING
# ============================================================================
print()
print('='*80)
print('[5] ABERRATION TESTING')
print('='*80)

if len(views) > 0:
    mean_v = np.mean(views)
    std_v = np.std(views)
    z_scores = (views - mean_v) / std_v

    o2 = np.sum(np.abs(z_scores) > 2)
    o3 = np.sum(np.abs(z_scores) > 3)
    o4 = np.sum(np.abs(z_scores) > 4)

    print(f'  Outlier Analysis (Z-score):')
    print(f'    |Z| > 2: {o2:,} ({o2/n*100:.2f}%) - Expected ~5%')
    print(f'    |Z| > 3: {o3:,} ({o3/n*100:.2f}%) - Expected ~0.3%')
    print(f'    |Z| > 4: {o4:,} ({o4/n*100:.2f}%) - Expected ~0.01%')

    # IQR
    q1, q3 = np.percentile(views, [25, 75])
    iqr = q3 - q1
    iqr_out = np.sum((views < q1-1.5*iqr) | (views > q3+1.5*iqr))

    print()
    print(f'  IQR Analysis:')
    print(f'    Q1: {q1:,.0f}')
    print(f'    Median: {np.median(views):,.0f}')
    print(f'    Q3: {q3:,.0f}')
    print(f'    IQR: {iqr:,.0f}')
    print(f'    IQR Outliers: {iqr_out:,} ({iqr_out/n*100:.2f}%)')

    # Shape
    kurt = stats.kurtosis(views)
    skew = stats.skew(views)
    print()
    print(f'  Shape Statistics:')
    print(f'    Skewness: {skew:.3f} (0 = symmetric)')
    print(f'    Kurtosis: {kurt:.3f} (0 = normal)')

    aberration_status = 'PASS' if o3/n < 0.02 else 'WARNING'
    print()
    print(f'  Aberration Status: {aberration_status}')

    results['tests']['aberration'] = {
        'status': aberration_status,
        'z2_outliers': int(o2),
        'z3_outliers': int(o3),
        'z4_outliers': int(o4),
        'z3_outliers_pct': float(o3/n*100),
        'iqr_outliers': int(iqr_out),
        'skewness': float(skew),
        'kurtosis': float(kurt)
    }
else:
    results['tests']['aberration'] = {'status': 'SKIP', 'reason': 'No views data'}

# ============================================================================
# SECTION 6: ANTI-CHEAT VALIDATION (WITH TEMPORAL FILTERING)
# ============================================================================
print()
print('='*80)
print('[6] ANTI-CHEAT VALIDATION (Temporal Filtered)')
print('='*80)

if len(abs_cols) > 0:
    # TEMPORAL FILTERING: Exclude wireframed future periods with null views
    # Only train/test on rows where views_computed is actually populated
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_year = datetime.now().year

    # Build temporal filter mask
    temporal_mask = bfd_pd['views_computed'].notna()

    # Add period_end filter if column exists
    if 'period_end' in bfd_pd.columns:
        period_mask = bfd_pd['period_end'].fillna('9999-12-31') <= current_date
        temporal_mask = temporal_mask & period_mask
        print(f'  Temporal filter: period_end <= {current_date}')

    # Add release status filter if column exists
    if 'status' in bfd_pd.columns:
        unreleased = ['Upcoming', 'Announced', 'In Production', 'Post Production']
        status_mask = ~bfd_pd['status'].isin(unreleased)
        temporal_mask = temporal_mask & status_mask
        print(f'  Status filter: excluding {unreleased}')

    # Add start_year filter if column exists
    if 'start_year' in bfd_pd.columns:
        year_mask = bfd_pd['start_year'].fillna(0) <= current_year
        temporal_mask = temporal_mask & year_mask
        print(f'  Year filter: start_year <= {current_year}')

    # Apply temporal filter
    bfd_filtered = bfd_pd[temporal_mask].copy()

    print(f'  Original rows: {len(bfd_pd):,}')
    print(f'  After temporal filter: {len(bfd_filtered):,}')
    print(f'  Excluded (future/wireframe): {len(bfd_pd) - len(bfd_filtered):,}')
    print()

    if len(bfd_filtered) > 1000:
        print(f'  Training quick model with {len(abs_cols)} features...')

        X = bfd_filtered[abs_cols].fillna(0)
        y = bfd_filtered['views_computed']

        # Quick train/test split
        np.random.seed(42)
        test_idx = np.random.choice(len(X), size=int(len(X)*0.2), replace=False)
        train_idx = np.array([i for i in range(len(X)) if i not in test_idx])

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Use XGBoost with GPU
        import xgboost as xgb

        print('  Training XGBoost (1350 trees, GPU)...')
        model = xgb.XGBRegressor(
            tree_method='gpu_hist', device='cuda', n_estimators=1350,
            max_depth=8, learning_rate=0.05, random_state=42, verbosity=0
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        # Calculate metrics - MAPE only on non-zero actuals to avoid division issues
        non_zero_mask = y_test > 0
        if non_zero_mask.sum() > 100:
            mape = np.mean(np.abs((y_test[non_zero_mask] - pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
        else:
            mape = np.mean(np.abs((y_test - pred) / np.maximum(y_test, 1))) * 100

        r2 = 1 - np.sum((y_test - pred)**2) / np.sum((y_test - np.mean(y_test))**2)

        print(f'  Results (on {len(bfd_filtered):,} temporally valid rows):')
        print(f'    R²: {r2:.4f}')
        print(f'    MAPE: {mape:.2f}%')
        print()

        # Anti-cheat bounds
        r2_valid = 0.30 <= r2 <= 0.90
        mape_valid = 5.0 <= mape <= 40.0

        print(f'  Anti-Cheat Thresholds:')
        print(f'    R² in [0.30, 0.90]: {r2:.4f} -> {"PASS" if r2_valid else "FAIL (possible data leakage)"}')
        print(f'    MAPE in [5%, 40%]: {mape:.2f}% -> {"PASS" if mape_valid else "FAIL (possible data leakage)"}')

        anti_cheat_status = 'PASS' if (r2_valid and mape_valid) else 'FAIL'

        results['tests']['anti_cheat'] = {
            'status': anti_cheat_status,
            'r2': float(r2),
            'mape': float(mape),
            'r2_valid': r2_valid,
            'mape_valid': mape_valid,
            'trees': 1350,
            'rows_used': len(bfd_filtered),
            'rows_excluded': len(bfd_pd) - len(bfd_filtered),
            'temporal_filter': 'Applied'
        }
    else:
        print(f'  SKIP: Insufficient rows after temporal filter ({len(bfd_filtered)})')
        results['tests']['anti_cheat'] = {'status': 'SKIP', 'reason': f'Insufficient rows after filter: {len(bfd_filtered)}'}
else:
    print('  SKIP: Insufficient features for model training')
    results['tests']['anti_cheat'] = {'status': 'SKIP', 'reason': 'Insufficient features'}

# ============================================================================
# SECTION 7: RELEASE VALIDATION
# ============================================================================
print()
print('='*80)
print('[7] RELEASE VALIDATION')
print('='*80)

current_year = datetime.now().year

if 'status' in bfd_pd.columns:
    unreleased_statuses = ['Upcoming', 'Announced', 'In Production', 'Post Production']
    unreleased_by_status = bfd_pd['status'].isin(unreleased_statuses).sum()
    print(f'  Unreleased by status: {unreleased_by_status:,}')
else:
    unreleased_by_status = 0
    print('  status column not found')

if 'start_year' in bfd_pd.columns:
    future_years = (bfd_pd['start_year'] > current_year).sum()
    print(f'  Future start_year (>{current_year}): {future_years:,}')
else:
    future_years = 0
    print('  start_year column not found')

results['tests']['release_validation'] = {
    'status': 'PASS',
    'unreleased_by_status': int(unreleased_by_status),
    'future_years': int(future_years),
    'current_year': current_year
}

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print()
print('='*80)
print('VALIDATION SUMMARY - BFD V24.01')
print('='*80)

print()
print('┌─────────────────────────────────────────────────────────────────────────┐')
print('│                        TEST RESULTS                                      │')
print('├─────────────────────────────────────────────────────────────────────────┤')

for test_name, test_result in results['tests'].items():
    status = test_result.get('status', 'UNKNOWN')
    color = '' if status == 'PASS' else ''
    print(f'│ {test_name:<25} │ {status:<8} │')

print('└─────────────────────────────────────────────────────────────────────────┘')

# Overall status
all_passed = all(t.get('status') in ['PASS', 'SKIP'] for t in results['tests'].values())
overall = 'PASS' if all_passed else 'REVIEW'

print()
print(f'OVERALL VALIDATION STATUS: {overall}')
print('='*80)

results['overall'] = overall
results['file_tested'] = 'BFD_V24.01.parquet'

output_file = f'/mnt/c/Users/RoyT6/Downloads/MAPIE/VALIDATION_V24.01_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\nResults saved: {output_file}')
