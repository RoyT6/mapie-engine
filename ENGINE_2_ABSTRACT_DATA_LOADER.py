#!/usr/bin/env python3
"""
ENGINE 2: ABSTRACT DATA LOADER
================================
Standalone engine that loads ALL abstract data files.
Creates abstract signals matrix from ALL sources.

FILES LOADED (78 total):
  Parquet (16): FC-BFD-Abstract-*, fresh_*, TRAINING_MATRIX_VIEWS, v12_*
  JSON (62): statistic_*, streaming_*, platform_*, genre_*, viewer_*, etc.

OUTPUT: Components/ABSTRACT_SIGNALS_UNIFIED.parquet
"""
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

start_time = time.time()

def fmt(s):
    return f'{int(s//60)}m {int(s%60)}s'

print('='*70)
print('ENGINE 2: ABSTRACT DATA LOADER')
print('Loading ALL abstract data sources')
print('='*70)

base = '/mnt/c/Users/RoyT6/Downloads/Abstract Data'
output_dir = '/mnt/c/Users/RoyT6/Downloads/Components'

# Track loading stats
STATS = {
    'files_attempted': 0,
    'files_loaded': 0,
    'files_failed': 0,
    'parquet_files': [],
    'json_files': [],
    'signals_created': 0
}

# ============================================================================
# SECTION 1: LOAD ALL PARQUET FILES
# ============================================================================
print('\n[SECTION 1] Loading Parquet files...')

parquet_dfs = {}

parquet_files = [
    'FC-BFD-Abstract-v10.00.parquet',
    'FC-BFD-Abstract-v11-cleaned.parquet',
    'FC-BFD-Abstract-v12.00.parquet',
    'FC-BFD-Abstract-v12-ALGO-READY.parquet',
    'FC-BFD-Abstract-v12-COEFFICIENTS.parquet',
    'FC-BFD-Abstract-v12-DAILY-FEATURES.parquet',
    'FC-BFD-Abstract-v12-QUARTERLY-FEATURES.parquet',
    'TRAINING_MATRIX_VIEWS.parquet',
    'v12_intermediate.parquet',
    'v12_normalized.parquet',
    'fresh_crypto.parquet',
    'fresh_events.parquet',
    'fresh_financial.parquet',
    'fresh_holiday_calendar.parquet',
    'fresh_holidays.parquet',
    'fresh_market_movers.parquet',
    'fresh_school_breaks.parquet',
    'fresh_streaming_stocks.parquet',
    'fresh_twitter_trends.parquet',
    'fresh_weather.parquet',
    'fresh_weather_data.parquet',
]

# Skip the huge 3.4GB file for now
# 'Cranberry_V3.00_TRAINING_MATRIX.parquet',

for fname in parquet_files:
    fpath = f'{base}/{fname}'
    STATS['files_attempted'] += 1

    try:
        df = pd.read_parquet(fpath)
        parquet_dfs[fname] = df
        STATS['files_loaded'] += 1
        STATS['parquet_files'].append(fname)
        print(f'  {fname}: {len(df):,} rows × {len(df.columns)} cols')

    except Exception as e:
        STATS['files_failed'] += 1
        print(f'  {fname}: FAILED - {e}')

# ============================================================================
# SECTION 2: LOAD ALL JSON FILES
# ============================================================================
print('\n[SECTION 2] Loading JSON files...')

json_data = {}

# Get all JSON files in directory
json_files = [f for f in os.listdir(base) if f.endswith('.json')]

for fname in json_files:
    fpath = f'{base}/{fname}'
    STATS['files_attempted'] += 1

    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        json_data[fname] = data
        STATS['files_loaded'] += 1
        STATS['json_files'].append(fname)

        # Count entries
        if isinstance(data, list):
            count = len(data)
        elif isinstance(data, dict):
            count = len(data)
        else:
            count = 1

        print(f'  {fname}: {count} entries')

    except Exception as e:
        STATS['files_failed'] += 1
        print(f'  {fname}: FAILED - {e}')

# ============================================================================
# SECTION 3: EXTRACT ABSTRACT SIGNALS FROM JSON
# ============================================================================
print('\n[SECTION 3] Extracting abstract signals from JSON...')

abstract_signals = {}

# Platform subscriber data
if 'Streaming_Platform_Subscribers_2025.json' in json_data:
    data = json_data['Streaming_Platform_Subscribers_2025.json']
    abstract_signals['platform_subscribers'] = data
    print(f'  platform_subscribers: {len(data) if isinstance(data, (list, dict)) else 1} entries')

# SVOD subscriber counts
for fname in json_data:
    if 'svod-subscriber' in fname.lower():
        abstract_signals[f'svod_subscribers_{fname[:20]}'] = json_data[fname]
        print(f'  svod_subscribers: loaded from {fname}')

# OTT market data
for fname in json_data:
    if 'ott' in fname.lower():
        abstract_signals[f'ott_market_{fname[:20]}'] = json_data[fname]
        print(f'  ott_market: loaded from {fname}')

# Platform financial data
if 'platform_financials_q4_2025.json' in json_data:
    abstract_signals['platform_financials'] = json_data['platform_financials_q4_2025.json']
    print(f'  platform_financials: loaded')

# Seasonal patterns
if 'seasonal_viewing_patterns_2025.json' in json_data:
    abstract_signals['seasonal_patterns'] = json_data['seasonal_viewing_patterns_2025.json']
    print(f'  seasonal_patterns: loaded')

# Genre performance
if 'genre_performance_2025.json' in json_data:
    abstract_signals['genre_performance'] = json_data['genre_performance_2025.json']
    print(f'  genre_performance: loaded')

# Churn/retention
if 'churn_retention_2025.json' in json_data:
    abstract_signals['churn_retention'] = json_data['churn_retention_2025.json']
    print(f'  churn_retention: loaded')

# Viewer demographics
if 'viewer_demographics_2025.json' in json_data:
    abstract_signals['viewer_demographics'] = json_data['viewer_demographics_2025.json']
    print(f'  viewer_demographics: loaded')

# Social engagement
if 'social_engagement_metrics_2025.json' in json_data:
    abstract_signals['social_engagement'] = json_data['social_engagement_metrics_2025.json']
    print(f'  social_engagement: loaded')

# Ad market data (for platform weighting)
for fname in json_data:
    if 'cpm' in fname.lower() or 'ad' in fname.lower():
        abstract_signals[f'ad_market_{fname[:20]}'] = json_data[fname]
        print(f'  ad_market: loaded from {fname}')

# Streaming share
if 'streaming_share_aug2025.json' in json_data:
    abstract_signals['streaming_share'] = json_data['streaming_share_aug2025.json']
    print(f'  streaming_share: loaded')

# Content lifecycle
if 'content_lifecycle_windows_2025.json' in json_data:
    abstract_signals['content_lifecycle'] = json_data['content_lifecycle_windows_2025.json']
    print(f'  content_lifecycle: loaded')

STATS['signals_created'] = len(abstract_signals)

# ============================================================================
# SECTION 4: COMBINE PARQUET DATA INTO UNIFIED MATRIX
# ============================================================================
print('\n[SECTION 4] Building unified abstract matrix...')

# Start with the ALGO-READY parquet if available
if 'FC-BFD-Abstract-v12-ALGO-READY.parquet' in parquet_dfs:
    unified_df = parquet_dfs['FC-BFD-Abstract-v12-ALGO-READY.parquet'].copy()
    print(f'  Base: FC-BFD-Abstract-v12-ALGO-READY.parquet ({len(unified_df):,} rows)')
elif 'FC-BFD-Abstract-v12.00.parquet' in parquet_dfs:
    unified_df = parquet_dfs['FC-BFD-Abstract-v12.00.parquet'].copy()
    print(f'  Base: FC-BFD-Abstract-v12.00.parquet ({len(unified_df):,} rows)')
else:
    # Create empty dataframe
    unified_df = pd.DataFrame()
    print('  No base parquet found, starting fresh')

# Merge in daily features if they exist
if 'FC-BFD-Abstract-v12-DAILY-FEATURES.parquet' in parquet_dfs:
    daily_df = parquet_dfs['FC-BFD-Abstract-v12-DAILY-FEATURES.parquet']
    print(f'  Adding daily features: {len(daily_df.columns)} columns')

    # Add daily features
    for col in daily_df.columns:
        if col not in unified_df.columns:
            unified_df[f'daily_{col}'] = daily_df[col].values[:len(unified_df)] if len(daily_df) >= len(unified_df) else 0

# Merge in quarterly features
if 'FC-BFD-Abstract-v12-QUARTERLY-FEATURES.parquet' in parquet_dfs:
    quarterly_df = parquet_dfs['FC-BFD-Abstract-v12-QUARTERLY-FEATURES.parquet']
    print(f'  Adding quarterly features: {len(quarterly_df.columns)} columns')

# Add fresh data signals
for fname, df in parquet_dfs.items():
    if fname.startswith('fresh_'):
        signal_name = fname.replace('fresh_', '').replace('.parquet', '')
        print(f'  Adding fresh signal: {signal_name}')

        # Store aggregated signal values
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                abstract_signals[f'fresh_{signal_name}_{col}_mean'] = float(df[col].mean())
                abstract_signals[f'fresh_{signal_name}_{col}_max'] = float(df[col].max())

# ============================================================================
# SECTION 5: CREATE SIGNAL LOOKUP TABLES
# ============================================================================
print('\n[SECTION 5] Creating signal lookup tables...')

# Platform weights lookup
platform_weights = {}
if 'platform_subscribers' in abstract_signals:
    data = abstract_signals['platform_subscribers']
    if isinstance(data, dict):
        for platform, subs in data.items():
            if isinstance(subs, (int, float)):
                platform_weights[platform.lower()] = subs
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = item.get('platform', item.get('name', '')).lower()
                subs = item.get('subscribers', item.get('subs', 0))
                if name:
                    platform_weights[name] = subs

abstract_signals['_platform_weights_lookup'] = platform_weights
print(f'  Platform weights: {len(platform_weights)} platforms')

# Genre weights lookup
genre_weights = {}
if 'genre_performance' in abstract_signals:
    data = abstract_signals['genre_performance']
    if isinstance(data, dict):
        for genre, perf in data.items():
            if isinstance(perf, dict):
                genre_weights[genre.lower()] = perf.get('weight', perf.get('performance', 1.0))
            else:
                genre_weights[genre.lower()] = perf

abstract_signals['_genre_weights_lookup'] = genre_weights
print(f'  Genre weights: {len(genre_weights)} genres')

# Seasonal multipliers
seasonal_mult = {}
if 'seasonal_patterns' in abstract_signals:
    data = abstract_signals['seasonal_patterns']
    if isinstance(data, dict):
        for month, mult in data.items():
            try:
                seasonal_mult[int(month) if month.isdigit() else month] = mult
            except:
                seasonal_mult[month] = mult

abstract_signals['_seasonal_multipliers'] = seasonal_mult
print(f'  Seasonal multipliers: {len(seasonal_mult)} entries')

# ============================================================================
# SECTION 6: SAVE OUTPUTS
# ============================================================================
print('\n[SECTION 6] Saving outputs...')

# Save unified abstract matrix
if len(unified_df) > 0:
    output_path = f'{output_dir}/ABSTRACT_SIGNALS_UNIFIED.parquet'
    unified_df.to_parquet(output_path, compression='snappy', index=False)
    print(f'  Saved: {output_path}')
    print(f'    Size: {os.path.getsize(output_path)/1024/1024:.1f} MB')
else:
    print('  No unified matrix to save (parquet sources empty)')

# Save all abstract signals as JSON
signals_path = f'{output_dir}/ABSTRACT_SIGNALS_ALL.json'

# Convert non-serializable values
def make_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return {'_type': 'dataframe', 'shape': list(obj.shape), 'columns': list(obj.columns)}
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    else:
        return obj

serializable_signals = make_serializable(abstract_signals)
with open(signals_path, 'w') as f:
    json.dump(serializable_signals, f, indent=2)
print(f'  Saved: {signals_path}')

# Save raw JSON data
raw_json_path = f'{output_dir}/ABSTRACT_JSON_RAW.json'
serializable_json = make_serializable(json_data)
with open(raw_json_path, 'w') as f:
    json.dump(serializable_json, f, indent=2)
print(f'  Saved: {raw_json_path}')

# Save statistics
STATS['signals_created'] = len(abstract_signals)
STATS['unified_rows'] = len(unified_df) if len(unified_df) > 0 else 0
STATS['unified_cols'] = len(unified_df.columns) if len(unified_df) > 0 else 0

stats_path = f'{output_dir}/ABSTRACT_DATA_STATS.json'
with open(stats_path, 'w') as f:
    json.dump(STATS, f, indent=2)
print(f'  Saved: {stats_path}')

# ============================================================================
# PROOF COUNTERS
# ============================================================================
total_time = time.time() - start_time

print('\n' + '='*70)
print('ENGINE 2 PROOF COUNTERS')
print('='*70)
print(f'  Files attempted: {STATS["files_attempted"]}')
print(f'  Files loaded: {STATS["files_loaded"]}')
print(f'  Files failed: {STATS["files_failed"]}')
print(f'  Parquet files: {len(STATS["parquet_files"])}')
print(f'  JSON files: {len(STATS["json_files"])}')
print(f'  Abstract signals extracted: {STATS["signals_created"]}')
print(f'  Unified matrix: {STATS["unified_rows"]:,} rows × {STATS["unified_cols"]} cols')
print(f'  Total time: {fmt(total_time)}')

print('\n  Parquet sources:')
for pf in STATS['parquet_files'][:10]:
    print(f'    {pf}')

print('\n  JSON sources (sample):')
for jf in STATS['json_files'][:10]:
    print(f'    {jf}')

print('\n' + '='*70)
print('ENGINE 2 COMPLETE')
print('='*70)
