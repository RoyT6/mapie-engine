#!/usr/bin/env python3
"""
ENGINE 1: TRAINING DATA LOADER
================================
Standalone engine that loads ALL training data files.
Creates a unified training matrix from ALL sources.

FILES LOADED (26 total):
  CSV (10): ETL_trueviews, AGGREGATED_*, INCOMING_*, NETFLIX_*, Netflix_*
  XLSX (4): What_We_Watched 2024, 2025, NETFLIX 20K, ESSENTIAL Top 1004
  JSON (5): TRUE VIEWS SOURCES, WatchTime_to_Views, country_weights, studios

OUTPUT: Components/TRAINING_MATRIX_UNIFIED.parquet
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
print('ENGINE 1: TRAINING DATA LOADER')
print('Loading ALL training data sources')
print('='*70)

base = '/mnt/c/Users/RoyT6/Downloads/Views Training Data'
output_dir = '/mnt/c/Users/RoyT6/Downloads/Components'

# Track loading stats
STATS = {
    'files_attempted': 0,
    'files_loaded': 0,
    'files_failed': 0,
    'total_records': 0,
    'total_views': 0,
    'sources': {}
}

# Master list to collect all records
all_records = []

def clean_views(val):
    """Clean views value to numeric"""
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).replace(',', '').replace(' ', '').strip()
    try:
        return float(val)
    except:
        return 0

def clean_hours(val):
    """Clean hours value to numeric"""
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).replace(',', '').replace(' ', '').strip()
    try:
        return float(val)
    except:
        return 0

# ============================================================================
# SECTION 1: CSV FILES
# ============================================================================
print('\n[SECTION 1] Loading CSV files...')

csv_files = [
    ('ETL_trueviews.csv', ['views', 'hours_viewed', 'title', 'runtime_minutes']),
    ('AGGREGATED_VIEWS_BY_IMDB.csv', None),
    ('AGGREGATED_VIEWS_BY_TITLE.csv', None),
    ('AGGREGATED_VIEWS_BY_TMDB.csv', None),
    ('INCOMING_hoursviewed_FP_ALL_tv.csv', ['views', 'title']),
    ('INCOMING_hoursviewed_FP_ALL_movies.csv', ['views', 'title']),
    ('INCOMING_hoursviewed_FP_29.25_tv_shows.csv', ['views', 'title']),
    ('INCOMING_metadata_FP_29.25_tv_shows.csv', None),
    ('NETFLIX 2023 2024 wGENRES.csv', None),
    ('Netflix_Master_First_Week_Hours.csv', None),
    ('Netflix_Most_Watched_By_Hours_Viewed.csv', None),
    ('Netflix_Movies_First_Week_Hours.csv', None),
]

for fname, expected_cols in csv_files:
    fpath = f'{base}/{fname}'
    STATS['files_attempted'] += 1

    try:
        df = pd.read_csv(fpath, low_memory=False)
        print(f'  {fname}: {len(df):,} rows × {len(df.columns)} cols')

        # Find views column
        views_col = None
        for col in df.columns:
            if 'views' in col.lower() and 'hours' not in col.lower():
                views_col = col
                break

        # Find hours column
        hours_col = None
        for col in df.columns:
            if 'hours' in col.lower():
                hours_col = col
                break

        # Find title column
        title_col = None
        for col in df.columns:
            if 'title' in col.lower():
                title_col = col
                break

        # For Netflix wGENRES, sum all Views columns
        if 'GENRES' in fname:
            views_cols = [c for c in df.columns if 'Views' in c]
            hours_cols = [c for c in df.columns if 'Hours' in c]

            for col in views_cols:
                df[col] = df[col].apply(clean_views)
            df['_total_views'] = df[views_cols].sum(axis=1)

            for col in hours_cols:
                df[col] = df[col].apply(clean_hours)
            df['_total_hours'] = df[hours_cols].sum(axis=1)

            views_col = '_total_views'
            hours_col = '_total_hours'

        # Extract records
        count_added = 0
        for _, row in df.iterrows():
            views = clean_views(row.get(views_col, 0)) if views_col else 0
            hours = clean_hours(row.get(hours_col, 0)) if hours_col else 0
            title = str(row.get(title_col, '')).strip() if title_col else ''

            if views > 0 or hours > 0:
                all_records.append({
                    'title': title,
                    'views': views,
                    'hours_viewed': hours,
                    'source_file': fname,
                    'source_type': 'csv'
                })
                count_added += 1

        STATS['files_loaded'] += 1
        STATS['sources'][fname] = count_added
        print(f'    → Added {count_added:,} records')

    except Exception as e:
        STATS['files_failed'] += 1
        print(f'  {fname}: FAILED - {e}')

# ============================================================================
# SECTION 2: EXCEL FILES
# ============================================================================
print('\n[SECTION 2] Loading XLSX files...')

xlsx_files = [
    'What_We_Watched_A_Netflix_Engagement_Report_2024Jan-Jun.xlsx',
    'What_We_Watched_A_Netflix_Engagement_Report_2025Jan-Jun.xlsx',
    'NETFLIX 20K Records 23-24.xlsx',
    'ESSENTIAL Top 1004 TV Shows Dec 2025.xlsx',
]

for fname in xlsx_files:
    fpath = f'{base}/{fname}'
    STATS['files_attempted'] += 1

    try:
        # Try reading first sheet
        df = pd.read_excel(fpath, sheet_name=0)
        print(f'  {fname}: {len(df):,} rows × {len(df.columns)} cols')

        # Find views/hours columns
        views_col = None
        hours_col = None
        title_col = None

        for col in df.columns:
            col_lower = str(col).lower()
            if 'view' in col_lower and 'hour' not in col_lower:
                views_col = col
            if 'hour' in col_lower:
                hours_col = col
            if 'title' in col_lower:
                title_col = col

        # Extract records
        count_added = 0
        for _, row in df.iterrows():
            views = clean_views(row.get(views_col, 0)) if views_col else 0
            hours = clean_hours(row.get(hours_col, 0)) if hours_col else 0
            title = str(row.get(title_col, '')).strip() if title_col else ''

            # If no views but hours, compute views estimate
            if views == 0 and hours > 0:
                views = hours * 2  # Rough estimate: 2 views per hour

            if views > 0 or hours > 0:
                all_records.append({
                    'title': title,
                    'views': views,
                    'hours_viewed': hours,
                    'source_file': fname,
                    'source_type': 'xlsx'
                })
                count_added += 1

        STATS['files_loaded'] += 1
        STATS['sources'][fname] = count_added
        print(f'    → Added {count_added:,} records')

    except Exception as e:
        STATS['files_failed'] += 1
        print(f'  {fname}: FAILED - {e}')

# ============================================================================
# SECTION 3: JSON CONFIG FILES
# ============================================================================
print('\n[SECTION 3] Loading JSON files...')

json_files = [
    'TRUE VIEWS SOURCES - 52.json',
    'WatchTime_to_Views_Config.json',
    'country_viewership_weights_2025.json',
    'top 55 studios.json',
    'top 50 most successful tv studios.json',
]

json_data = {}
for fname in json_files:
    fpath = f'{base}/{fname}'
    STATS['files_attempted'] += 1

    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        json_data[fname] = data
        STATS['files_loaded'] += 1

        # Count entries
        if isinstance(data, list):
            count = len(data)
        elif isinstance(data, dict):
            count = len(data)
        else:
            count = 1

        STATS['sources'][fname] = count
        print(f'  {fname}: {count} entries (config/reference)')

    except Exception as e:
        STATS['files_failed'] += 1
        print(f'  {fname}: FAILED - {e}')

# ============================================================================
# SECTION 4: IMDB SEASONS ALLOCATED (Large file)
# ============================================================================
print('\n[SECTION 4] Loading IMDB Seasons Allocated...')

try:
    STATS['files_attempted'] += 1
    fname = 'IMDB Seasons Allocated.csv'
    fpath = f'{base}/{fname}'

    # This is a large file, read in chunks
    chunk_size = 100000
    count_added = 0

    for chunk in pd.read_csv(fpath, low_memory=False, chunksize=chunk_size):
        # Find columns
        views_col = None
        title_col = None
        imdb_col = None

        for col in chunk.columns:
            col_lower = str(col).lower()
            if 'view' in col_lower:
                views_col = col
            if 'title' in col_lower or 'name' in col_lower:
                title_col = col
            if 'imdb' in col_lower or 'tconst' in col_lower:
                imdb_col = col

        for _, row in chunk.iterrows():
            views = clean_views(row.get(views_col, 0)) if views_col else 0
            title = str(row.get(title_col, '')).strip() if title_col else ''
            imdb = str(row.get(imdb_col, '')).strip() if imdb_col else ''

            if views > 0:
                all_records.append({
                    'title': title,
                    'views': views,
                    'hours_viewed': 0,
                    'imdb_id': imdb,
                    'source_file': fname,
                    'source_type': 'csv_large'
                })
                count_added += 1

    STATS['files_loaded'] += 1
    STATS['sources'][fname] = count_added
    print(f'  {fname}: Added {count_added:,} records')

except Exception as e:
    STATS['files_failed'] += 1
    print(f'  IMDB Seasons Allocated.csv: FAILED - {e}')

# ============================================================================
# SECTION 5: BUILD UNIFIED TRAINING MATRIX
# ============================================================================
print('\n[SECTION 5] Building unified training matrix...')

training_df = pd.DataFrame(all_records)

# Add computed columns
training_df['title_normalized'] = training_df['title'].str.lower().str.strip()
training_df['has_views'] = (training_df['views'] > 0).astype(int)
training_df['has_hours'] = (training_df['hours_viewed'] > 0).astype(int)

# Log transform
training_df['views_log'] = np.log1p(training_df['views'])
training_df['hours_log'] = np.log1p(training_df['hours_viewed'])

print(f'  Total records: {len(training_df):,}')
print(f'  Total views: {training_df["views"].sum():,.0f}')
print(f'  Total hours: {training_df["hours_viewed"].sum():,.0f}')

# Distribution stats
print(f'\n  Views distribution:')
for p in [10, 25, 50, 75, 90, 95, 99]:
    val = training_df['views'].quantile(p/100)
    print(f'    P{p}: {val:,.0f}')

# ============================================================================
# SECTION 6: SAVE OUTPUT
# ============================================================================
print('\n[SECTION 6] Saving outputs...')

# Save training matrix
output_path = f'{output_dir}/TRAINING_MATRIX_UNIFIED.parquet'
training_df.to_parquet(output_path, compression='snappy', index=False)
print(f'  Saved: {output_path}')
print(f'    Size: {os.path.getsize(output_path)/1024/1024:.1f} MB')

# Save statistics
STATS['total_records'] = len(training_df)
STATS['total_views'] = float(training_df['views'].sum())
STATS['distribution'] = {
    'min': float(training_df['views'].min()),
    'max': float(training_df['views'].max()),
    'mean': float(training_df['views'].mean()),
    'median': float(training_df['views'].median()),
    'std': float(training_df['views'].std()),
}

stats_path = f'{output_dir}/TRAINING_DATA_STATS.json'
with open(stats_path, 'w') as f:
    json.dump(STATS, f, indent=2)
print(f'  Saved: {stats_path}')

# Save JSON configs separately for easy access
configs_path = f'{output_dir}/TRAINING_CONFIGS.json'
with open(configs_path, 'w') as f:
    json.dump(json_data, f, indent=2)
print(f'  Saved: {configs_path}')

# ============================================================================
# PROOF COUNTERS
# ============================================================================
total_time = time.time() - start_time

print('\n' + '='*70)
print('ENGINE 1 PROOF COUNTERS')
print('='*70)
print(f'  Files attempted: {STATS["files_attempted"]}')
print(f'  Files loaded: {STATS["files_loaded"]}')
print(f'  Files failed: {STATS["files_failed"]}')
print(f'  Total records: {len(training_df):,}')
print(f'  Total views: {training_df["views"].sum():,.0f}')
print(f'  Unique titles: {training_df["title_normalized"].nunique():,}')
print(f'  Sources:')
for src, cnt in sorted(STATS['sources'].items(), key=lambda x: -x[1])[:10]:
    print(f'    {src}: {cnt:,}')
print(f'  Total time: {fmt(total_time)}')

print('\n' + '='*70)
print('ENGINE 1 COMPLETE')
print('='*70)
