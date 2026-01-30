#!/usr/bin/env python3
"""
MAPIE MULTI-FILE VALIDATION
============================
Validates BFD_V27.00.parquet (or fallback to V26/V24)
Schema V27.00 compliance with monthly/weekly temporal granularity
Provides comparison and suggestions
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
print('MAPIE MULTI-FILE VALIDATION')
print('='*80)
print(f'Timestamp: {datetime.now().isoformat()}')
print(f'GPU: {cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}')
print()

BASE = '/mnt/c/Users/RoyT6/Downloads'

results = {
    'timestamp': datetime.now().isoformat(),
    'files': {},
    'comparison': {},
    'suggestions': []
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def clear_gpu():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def fmt_bytes(b):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024:
            return f'{b:.1f} {unit}'
        b /= 1024
    return f'{b:.1f} TB'

# ============================================================================
# FILE 1: BFD_V27.00.parquet (or fallback to V26/V24)
# ============================================================================
print('='*80)
print('[FILE 1] BFD Database (V27.00 preferred)')
print('='*80)

# V27.00 compliance: Check for latest version first
BFD_V27_PATH = f'{BASE}/BFD_V27.00.parquet'
BFD_V26_PATH = f'{BASE}/BFD_V26.00.parquet'
BFD_V24_PATH = f'{BASE}/BFD_V24.02.parquet'
if os.path.exists(BFD_V27_PATH):
    BFD_PATH = BFD_V27_PATH
    bfd_results = {'file': 'BFD_V27.00.parquet', 'schema_version': '27.00'}
elif os.path.exists(BFD_V26_PATH):
    BFD_PATH = BFD_V26_PATH
    bfd_results = {'file': 'BFD_V26.00.parquet', 'schema_version': '26.00', 'upgrade_needed': True}
    print('  WARNING: V27.00 not found, falling back to V26.00')
else:
    BFD_PATH = BFD_V24_PATH
    bfd_results = {'file': 'BFD_V24.02.parquet', 'schema_version': '24.02', 'upgrade_needed': True}
    print('  WARNING: V27.00/V26.00 not found, falling back to V24.02')

try:
    t0 = datetime.now()
    file_size = os.path.getsize(BFD_PATH)
    print(f'  File size: {fmt_bytes(file_size)}')

    bfd = cudf.read_parquet(BFD_PATH)
    load_time = (datetime.now() - t0).total_seconds()

    bfd_pd = bfd.to_pandas()
    del bfd
    clear_gpu()

    print(f'  Rows: {len(bfd_pd):,}')
    print(f'  Columns: {len(bfd_pd.columns)}')
    print(f'  Load time: {load_time:.2f}s')

    bfd_results['rows'] = len(bfd_pd)
    bfd_results['columns'] = len(bfd_pd.columns)
    bfd_results['file_size_bytes'] = file_size
    bfd_results['load_time'] = load_time

    # Schema check
    print('\n  Schema Analysis:')
    required = ['views_computed', 'title', 'fc_uid']
    optional = ['imdb_id', 'tmdb_id', 'genres', 'title_type', 'start_year', 'status', 'period_end']

    # V27.00 Temporal Granularity columns (replaces Nielsen branding)
    v27_temporal_cols = [
        'minutes_viewed_millions', 'avg_view_duration_min', 'streaming_platform_primary',
        'period_start', 'period_end', 'views_weekly_total', 'views_weekly_us', 'views_weekly_cn',
        'views_weekly_in', 'views_weekly_gb', 'views_weekly_br', 'views_weekly_de',
        'views_weekly_jp', 'views_weekly_fr', 'views_weekly_ca', 'views_weekly_mx',
        'views_weekly_au', 'views_weekly_es', 'views_weekly_it', 'views_weekly_kr', 'views_weekly_row'
    ]

    # V27.00 NEW: Monthly/Weekly temporal patterns
    v27_monthly_pattern_cols = [c for c in bfd_pd.columns if c.startswith('views_m') or c.startswith('hours_m')]
    v27_weekly_pattern_cols = [c for c in bfd_pd.columns if c.startswith('views_w') or c.startswith('hours_w')]

    # V27.00 NEW: Parrot Analytics columns
    v27_parrot_cols = [c for c in bfd_pd.columns if c.startswith('parrot_')]

    # V27.00 NEW: Hours columns
    v27_hours_cols = [c for c in bfd_pd.columns if c.startswith('hours_') and not c.startswith('hours_m') and not c.startswith('hours_w')]

    abs_cols = [c for c in bfd_pd.columns if c.startswith('abs_')]
    views_cols = [c for c in bfd_pd.columns if c.startswith('views_')]
    wvv_cols = [c for c in bfd_pd.columns if c.startswith('wvv_')]
    views_weekly_cols = [c for c in bfd_pd.columns if c.startswith('views_weekly_')]

    found_required = [c for c in required if c in bfd_pd.columns]
    found_optional = [c for c in optional if c in bfd_pd.columns]
    found_v27_temporal = [c for c in v27_temporal_cols if c in bfd_pd.columns]

    print(f'    Required columns: {len(found_required)}/{len(required)}')
    print(f'    Optional columns: {len(found_optional)}/{len(optional)}')
    print(f'    Abstract signals: {len(abs_cols)}')
    print(f'    Views columns: {len(views_cols)}')
    print(f'    WVV columns: {len(wvv_cols)}')
    print(f'    V27 Temporal columns: {len(found_v27_temporal)}/{len(v27_temporal_cols)}')
    print(f'    Views Weekly columns (actual): {len(views_weekly_cols)}')
    print(f'    V27 Monthly pattern columns: {len(v27_monthly_pattern_cols)}')
    print(f'    V27 Weekly pattern columns: {len(v27_weekly_pattern_cols)}')
    print(f'    V27 Hours columns: {len(v27_hours_cols)}')
    print(f'    V27 Parrot columns: {len(v27_parrot_cols)}')

    # V27.00 compliance check - requires temporal patterns OR legacy columns
    v27_compliant = len(found_v27_temporal) >= 15 or len(v27_monthly_pattern_cols) > 0 or len(v27_weekly_pattern_cols) > 0
    print(f'    V27.00 Temporal Compliant: {"YES" if v27_compliant else "NO - UPGRADE NEEDED"}')

    bfd_results['schema'] = {
        'required_found': len(found_required),
        'required_missing': [c for c in required if c not in bfd_pd.columns],
        'optional_found': len(found_optional),
        'abstract_signals': len(abs_cols),
        'views_columns': len(views_cols),
        'wvv_columns': len(wvv_cols),
        'v27_temporal_found': len(found_v27_temporal),
        'v27_temporal_missing': [c for c in v27_temporal_cols if c not in bfd_pd.columns],
        'views_weekly_columns': len(views_weekly_cols),
        'v27_monthly_pattern_columns': len(v27_monthly_pattern_cols),
        'v27_weekly_pattern_columns': len(v27_weekly_pattern_cols),
        'v27_hours_columns': len(v27_hours_cols),
        'v27_parrot_columns': len(v27_parrot_cols),
        'v27_compliant': v27_compliant
    }

    # Data quality
    print('\n  Data Quality:')
    if 'views_computed' in bfd_pd.columns:
        vc = bfd_pd['views_computed']
        nulls = vc.isna().sum()
        zeros = (vc == 0).sum()
        negatives = (vc < 0).sum()

        # Filter for valid data
        valid = vc.dropna()
        if len(valid) > 0:
            print(f'    views_computed mean: {valid.mean():,.0f}')
            print(f'    views_computed median: {valid.median():,.0f}')
            print(f'    views_computed min: {valid.min():,.0f}')
            print(f'    views_computed max: {valid.max():,.0f}')

        print(f'    Nulls: {nulls:,} ({nulls/len(bfd_pd)*100:.2f}%)')
        print(f'    Zeros: {zeros:,} ({zeros/len(bfd_pd)*100:.2f}%)')
        print(f'    Negatives: {negatives:,}')

        bfd_results['views_computed'] = {
            'nulls': int(nulls),
            'null_pct': float(nulls/len(bfd_pd)*100),
            'zeros': int(zeros),
            'negatives': int(negatives),
            'mean': float(valid.mean()) if len(valid) > 0 else None,
            'median': float(valid.median()) if len(valid) > 0 else None,
            'min': float(valid.min()) if len(valid) > 0 else None,
            'max': float(valid.max()) if len(valid) > 0 else None
        }

    # Temporal analysis
    print('\n  Temporal Analysis:')
    current_year = datetime.now().year

    if 'status' in bfd_pd.columns:
        unreleased = ['Upcoming', 'Announced', 'In Production', 'Post Production']
        unreleased_count = bfd_pd['status'].isin(unreleased).sum()
        print(f'    Unreleased (by status): {unreleased_count:,}')
        bfd_results['unreleased_status'] = int(unreleased_count)

    if 'start_year' in bfd_pd.columns:
        future = (bfd_pd['start_year'] > current_year).sum()
        print(f'    Future start_year (>{current_year}): {future:,}')
        bfd_results['future_years'] = int(future)

    # Anti-cheat validation with temporal filter
    print('\n  Anti-Cheat Validation (Temporal Filtered):')
    if len(abs_cols) > 0 and 'views_computed' in bfd_pd.columns:
        # Apply temporal filter
        mask = bfd_pd['views_computed'].notna()
        if 'status' in bfd_pd.columns:
            mask = mask & ~bfd_pd['status'].isin(['Upcoming', 'Announced', 'In Production', 'Post Production'])
        if 'start_year' in bfd_pd.columns:
            mask = mask & (bfd_pd['start_year'].fillna(0) <= current_year)

        filtered = bfd_pd[mask]
        print(f'    Rows after filter: {len(filtered):,}')

        if len(filtered) > 1000:
            X = filtered[abs_cols].fillna(0)
            y = filtered['views_computed']

            np.random.seed(42)
            test_idx = np.random.choice(len(X), size=int(len(X)*0.2), replace=False)
            train_idx = np.array([i for i in range(len(X)) if i not in test_idx])

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            import xgboost as xgb
            model = xgb.XGBRegressor(
                tree_method='gpu_hist', device='cuda', n_estimators=1350,
                max_depth=8, learning_rate=0.05, random_state=42, verbosity=0
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            non_zero = y_test > 0
            if non_zero.sum() > 100:
                mape = np.mean(np.abs((y_test[non_zero] - pred[non_zero]) / y_test[non_zero])) * 100
            else:
                mape = np.mean(np.abs((y_test - pred) / np.maximum(y_test, 1))) * 100

            r2 = 1 - np.sum((y_test - pred)**2) / np.sum((y_test - np.mean(y_test))**2)

            r2_valid = 0.30 <= r2 <= 0.90
            mape_valid = 5.0 <= mape <= 40.0

            print(f'    R²: {r2:.4f} ({"PASS" if r2_valid else "FAIL"})')
            print(f'    MAPE: {mape:.2f}% ({"PASS" if mape_valid else "FAIL"})')

            bfd_results['anti_cheat'] = {
                'r2': float(r2),
                'mape': float(mape),
                'r2_valid': r2_valid,
                'mape_valid': mape_valid,
                'status': 'PASS' if (r2_valid and mape_valid) else 'FAIL'
            }

    bfd_results['status'] = 'LOADED'

except Exception as e:
    print(f'  ERROR: {e}')
    bfd_results['status'] = 'ERROR'
    bfd_results['error'] = str(e)

results['files']['bfd_v24_02'] = bfd_results
clear_gpu()

# ============================================================================
# FILE 2: BFD_Star_Schema_V24.00.parquet
# ============================================================================
print()
print('='*80)
print('[FILE 2] BFD_Star_Schema_V24.00.parquet')
print('='*80)

STAR_PATH = f'{BASE}/BFD_Star_Schema_V24.00.parquet'
star_results = {'file': 'BFD_Star_Schema_V24.00.parquet'}

try:
    t0 = datetime.now()
    file_size = os.path.getsize(STAR_PATH)
    print(f'  File size: {fmt_bytes(file_size)}')

    star = cudf.read_parquet(STAR_PATH)
    load_time = (datetime.now() - t0).total_seconds()

    star_pd = star.to_pandas()
    del star
    clear_gpu()

    print(f'  Rows: {len(star_pd):,}')
    print(f'  Columns: {len(star_pd.columns)}')
    print(f'  Load time: {load_time:.2f}s')

    star_results['rows'] = len(star_pd)
    star_results['columns'] = len(star_pd.columns)
    star_results['file_size_bytes'] = file_size
    star_results['load_time'] = load_time

    # Schema check for Star Schema
    print('\n  Schema Analysis:')
    print(f'    Columns: {list(star_pd.columns)}')

    expected_star_cols = ['fc_uid', 'country', 'platform', 'views']
    found_star = [c for c in expected_star_cols if c in star_pd.columns]
    print(f'    Expected Star Schema columns: {len(found_star)}/{len(expected_star_cols)}')

    star_results['schema'] = {
        'columns': list(star_pd.columns),
        'expected_found': len(found_star),
        'expected_missing': [c for c in expected_star_cols if c not in star_pd.columns]
    }

    # Dimension analysis
    print('\n  Dimension Analysis:')

    if 'fc_uid' in star_pd.columns:
        unique_titles = star_pd['fc_uid'].nunique()
        print(f'    Unique titles (fc_uid): {unique_titles:,}')
        star_results['unique_titles'] = int(unique_titles)

    if 'country' in star_pd.columns:
        unique_countries = star_pd['country'].nunique()
        countries = star_pd['country'].unique().tolist()
        print(f'    Unique countries: {unique_countries}')
        print(f'    Countries: {countries[:10]}{"..." if len(countries) > 10 else ""}')
        star_results['unique_countries'] = int(unique_countries)
        star_results['countries'] = countries

    if 'platform' in star_pd.columns:
        unique_platforms = star_pd['platform'].nunique()
        platforms = star_pd['platform'].value_counts().head(20)
        print(f'    Unique platforms: {unique_platforms}')
        print(f'    Top platforms:')
        for p, c in platforms.items():
            print(f'      {p}: {c:,}')
        star_results['unique_platforms'] = int(unique_platforms)
        star_results['top_platforms'] = platforms.to_dict()

    # Views analysis
    print('\n  Views Analysis:')
    views_col = 'views' if 'views' in star_pd.columns else None
    if views_col is None:
        for c in star_pd.columns:
            if 'view' in c.lower():
                views_col = c
                break

    if views_col:
        views = star_pd[views_col]
        nulls = views.isna().sum()
        zeros = (views == 0).sum()
        negatives = (views < 0).sum()
        valid = views.dropna()

        print(f'    Views column: {views_col}')
        print(f'    Total views: {valid.sum():,.0f}')
        if len(valid) > 0:
            print(f'    Mean: {valid.mean():,.0f}')
            print(f'    Median: {valid.median():,.0f}')
        print(f'    Nulls: {nulls:,} ({nulls/len(star_pd)*100:.2f}%)')
        print(f'    Zeros: {zeros:,} ({zeros/len(star_pd)*100:.2f}%)')
        print(f'    Negatives: {negatives:,}')

        star_results['views'] = {
            'column': views_col,
            'total': float(valid.sum()),
            'mean': float(valid.mean()) if len(valid) > 0 else None,
            'nulls': int(nulls),
            'null_pct': float(nulls/len(star_pd)*100),
            'zeros': int(zeros),
            'negatives': int(negatives)
        }

    # Integrity checks
    print('\n  Integrity Checks:')

    # Check for duplicates
    if 'fc_uid' in star_pd.columns and 'country' in star_pd.columns and 'platform' in star_pd.columns:
        dupe_check = star_pd.groupby(['fc_uid', 'country', 'platform']).size()
        dupes = (dupe_check > 1).sum()
        print(f'    Duplicate (fc_uid, country, platform): {dupes:,}')
        star_results['duplicates'] = int(dupes)

    # Check views distribution by country
    if 'country' in star_pd.columns and views_col:
        country_views = star_pd.groupby('country')[views_col].sum().sort_values(ascending=False)
        print(f'    Views by country (top 5):')
        for country, v in country_views.head(5).items():
            pct = v / country_views.sum() * 100
            print(f'      {country}: {v:,.0f} ({pct:.1f}%)')
        star_results['views_by_country'] = country_views.to_dict()

    star_results['status'] = 'LOADED'

except Exception as e:
    print(f'  ERROR: {e}')
    star_results['status'] = 'ERROR'
    star_results['error'] = str(e)

results['files']['star_schema_v24_00'] = star_results

# ============================================================================
# COMPARISON & SUGGESTIONS
# ============================================================================
print()
print('='*80)
print('COMPARISON & SUGGESTIONS')
print('='*80)

suggestions = []

# Version alignment check
print('\n[1] Version Alignment:')
print(f'    BFD version: V24.02')
print(f'    Star Schema version: V24.00')
if True:  # Version mismatch
    suggestion = 'VERSION MISMATCH: Star Schema is V24.00 but BFD is V24.02. Consider regenerating Star Schema to match BFD version.'
    print(f'    ⚠️  {suggestion}')
    suggestions.append({'type': 'warning', 'category': 'version', 'message': suggestion})

# Row count comparison
print('\n[2] Row Count Analysis:')
if 'bfd_v24_02' in results['files'] and 'star_schema_v24_00' in results['files']:
    bfd_rows = results['files']['bfd_v24_02'].get('rows', 0)
    star_rows = results['files']['star_schema_v24_00'].get('rows', 0)
    star_titles = results['files']['star_schema_v24_00'].get('unique_titles', 0)

    print(f'    BFD rows: {bfd_rows:,}')
    print(f'    Star Schema rows: {star_rows:,}')
    print(f'    Star Schema unique titles: {star_titles:,}')

    if star_titles > 0:
        expansion_factor = star_rows / star_titles
        print(f'    Expansion factor: {expansion_factor:.1f}x (rows per title)')

    if star_titles > 0 and bfd_rows > 0:
        title_coverage = star_titles / bfd_rows * 100
        print(f'    Title coverage: {title_coverage:.1f}%')

        if title_coverage < 90:
            suggestion = f'LOW COVERAGE: Star Schema only covers {title_coverage:.1f}% of BFD titles. Consider full regeneration.'
            suggestions.append({'type': 'warning', 'category': 'coverage', 'message': suggestion})

# Data quality comparison
print('\n[3] Data Quality:')
if 'bfd_v24_02' in results['files']:
    bfd_data = results['files']['bfd_v24_02']
    if 'views_computed' in bfd_data:
        vc = bfd_data['views_computed']
        print(f'    BFD null views: {vc.get("null_pct", 0):.2f}%')

        if vc.get('null_pct', 0) > 10:
            suggestion = f'HIGH NULL RATE: {vc.get("null_pct", 0):.1f}% of views_computed is null. Verify temporal wireframe periods.'
            suggestions.append({'type': 'info', 'category': 'data_quality', 'message': suggestion})

if 'star_schema_v24_00' in results['files']:
    star_data = results['files']['star_schema_v24_00']
    if 'views' in star_data:
        sv = star_data['views']
        print(f'    Star Schema null views: {sv.get("null_pct", 0):.2f}%')
        print(f'    Star Schema zero views: {sv.get("zeros", 0):,}')

# Anti-cheat summary
print('\n[4] Anti-Cheat Validation:')
if 'bfd_v24_02' in results['files'] and 'anti_cheat' in results['files']['bfd_v24_02']:
    ac = results['files']['bfd_v24_02']['anti_cheat']
    print(f'    BFD V24.02: R²={ac.get("r2", 0):.4f}, MAPE={ac.get("mape", 0):.2f}% -> {ac.get("status", "N/A")}')

# Platform/Country integrity
print('\n[5] Star Schema Integrity:')
if 'star_schema_v24_00' in results['files']:
    star_data = results['files']['star_schema_v24_00']
    countries = star_data.get('unique_countries', 0)
    platforms = star_data.get('unique_platforms', 0)

    print(f'    Countries: {countries}')
    print(f'    Platforms: {platforms}')

    if countries != 18:
        suggestion = f'COUNTRY COUNT: Expected 18 countries, found {countries}. Verify country allocation.'
        suggestions.append({'type': 'warning', 'category': 'schema', 'message': suggestion})

# Recommendations
print('\n' + '='*80)
print('RECOMMENDATIONS')
print('='*80)

if not suggestions:
    print('\n  ✅ No critical issues found.')
else:
    for i, s in enumerate(suggestions, 1):
        icon = '⚠️' if s['type'] == 'warning' else 'ℹ️'
        print(f'\n  {i}. {icon} [{s["category"].upper()}] {s["message"]}')

results['suggestions'] = suggestions

# Additional recommendations
print('\n  Additional Recommendations:')
print('  1. Regenerate Star Schema V24.02 to match BFD version')
print('  2. Run MAPIE_INTEGRATED_RUNNER.py to optimize weights for V24.02')
print('  3. Verify conservation rule: SUM(country views) = total views per title')

# ============================================================================
# SUMMARY
# ============================================================================
print()
print('='*80)
print('VALIDATION SUMMARY')
print('='*80)

print('\n┌────────────────────────────────────────────────────────────────────────────┐')
print('│                           FILE STATUS                                       │')
print('├────────────────────────────────────────────────────────────────────────────┤')

for name, data in results['files'].items():
    status = data.get('status', 'UNKNOWN')
    rows = data.get('rows', 0)
    cols = data.get('columns', 0)
    print(f'│ {name:<25} │ {status:<8} │ {rows:>12,} rows │ {cols:>5} cols │')

print('└────────────────────────────────────────────────────────────────────────────┘')

overall = 'PASS' if all(f.get('status') == 'LOADED' for f in results['files'].values()) else 'REVIEW'
print(f'\nOVERALL STATUS: {overall}')
print(f'SUGGESTIONS: {len(suggestions)}')

# Save results
output_file = f'/mnt/c/Users/RoyT6/Downloads/MAPIE/VALIDATION_MULTI_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\nResults saved: {output_file}')
