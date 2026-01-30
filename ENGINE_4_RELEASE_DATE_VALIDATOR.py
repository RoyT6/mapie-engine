#!/usr/bin/env python3
"""
ENGINE 4: RELEASE DATE VALIDATOR
=================================
Standalone engine that validates premiere dates for TV seasons.
Scrapes TMDB API for per-season air dates.
Blocks views calculation for unreleased content.

PROBLEM SOLVED:
  TV seasons like "Reacher Season 4" have premiere_date = series premiere (2022-02-03)
  instead of the actual season premiere date. This causes unreleased seasons to get views.

SOLUTION:
  1. Load BFD database
  2. For TV seasons, call TMDB API: /tv/{tmdb_id}/season/{season_number}
  3. Get actual air_date for each season
  4. Flag records where season hasn't aired yet
  5. Output validated database with correct premiere dates

OUTPUT: Cranberry_BFD_RELEASE_VALIDATED_{date}_V{version}.parquet
"""
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()

def fmt(s):
    return f'{int(s//60)}m {int(s%60)}s'

print('='*70)
print('ENGINE 4: RELEASE DATE VALIDATOR')
print('Validating premiere dates for TV seasons via TMDB API')
print('='*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
import platform
if platform.system() == 'Windows':
    BASE_DIR = r'C:\Users\RoyT6\Downloads'
else:
    BASE_DIR = '/mnt/c/Users/RoyT6/Downloads'
OUTPUT_DIR = BASE_DIR

# TMDB API Configuration
TMDB_API_KEY = '8fe4d782052d6c6dce4cf23f4b01b3c5'
TMDB_BEARER_TOKEN = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI4ZmU0ZDc4MjA1MmQ2YzZkY2U0Y2YyM2Y0YjAxYjNjNSIsIm5iZiI6MTc1OTk3NzE1Ny4zMTEsInN1YiI6IjY4ZTcxZWM1MjdjZjcxOTcyNDczZDc5NiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ._Q_5lA7npuYpFJzYOFAsvxdGjzzKZSoCOt-63BJ8m4U'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

# Today's date for comparison
TODAY = date.today()
TODAY_STR = TODAY.strftime('%Y-%m-%d')

# Version tracking
RUN_DATE = datetime.now().strftime('%Y%m%d')
VERSION = '4.12'

# Rate limiting
API_DELAY = 0.05  # 50ms between requests to avoid rate limits
MAX_WORKERS = 5   # Parallel API calls

# Stats tracking
STATS = {
    'total_records': 0,
    'tv_seasons': 0,
    'films': 0,
    'tmdb_lookups_attempted': 0,
    'tmdb_lookups_success': 0,
    'tmdb_lookups_failed': 0,
    'dates_updated': 0,
    'unreleased_flagged': 0,
    'views_blocked': 0,
    'views_blocked_total': 0
}

# ============================================================================
# SECTION 1: LOAD DATABASE
# ============================================================================
print('\n[SECTION 1] Loading database...')

# Find the latest BFD file
bfd_files = [f for f in os.listdir(BASE_DIR) if f.startswith('Cranberry_BFD_MAPIE') and f.endswith('.parquet')]
if bfd_files:
    bfd_files.sort(reverse=True)
    input_file = f'{BASE_DIR}/{bfd_files[0]}'
else:
    # Fall back to computed file
    input_file = f'{BASE_DIR}/Cranberry_COMPUTED_V4.11.parquet'

print(f'  Loading: {input_file}')
df = pd.read_parquet(input_file)
print(f'  Loaded: {len(df):,} rows × {len(df.columns)} columns')

STATS['total_records'] = len(df)

# ============================================================================
# SECTION 2: IDENTIFY TV SEASONS
# ============================================================================
print('\n[SECTION 2] Identifying TV seasons...')

# Check what columns we have
print(f'  Columns available: {len(df.columns)}')

# Find title_type column
title_type_col = None
for col in ['title_type', 'type', 'content_type', 'media_type']:
    if col in df.columns:
        title_type_col = col
        break

if title_type_col:
    print(f'  Using title type column: {title_type_col}')
    type_counts = df[title_type_col].value_counts()
    print(f'  Type distribution:')
    for t, c in type_counts.items():
        print(f'    {t}: {c:,}')

    # Identify TV seasons
    tv_mask = df[title_type_col].isin(['tv_season', 'tv', 'series', 'TV Season', 'TV Series'])
    tv_seasons = df[tv_mask].copy()
    STATS['tv_seasons'] = len(tv_seasons)
    STATS['films'] = len(df) - len(tv_seasons)
    print(f'  TV seasons identified: {STATS["tv_seasons"]:,}')
else:
    print('  No title_type column found, checking for season_number')
    if 'season_number' in df.columns:
        tv_mask = df['season_number'].notna() & (df['season_number'] > 0)
        tv_seasons = df[tv_mask].copy()
        STATS['tv_seasons'] = len(tv_seasons)
        print(f'  TV seasons (by season_number): {STATS["tv_seasons"]:,}')
    else:
        tv_seasons = pd.DataFrame()
        STATS['tv_seasons'] = 0
        print('  No TV season identification possible')

# ============================================================================
# SECTION 3: TMDB API FUNCTIONS
# ============================================================================
print('\n[SECTION 3] Setting up TMDB API...')

# Headers for API calls
HEADERS = {
    'Authorization': f'Bearer {TMDB_BEARER_TOKEN}',
    'Content-Type': 'application/json'
}

def get_season_info(tmdb_id, season_number):
    """
    Fetch season info from TMDB API.
    Returns air_date for the season.
    """
    if pd.isna(tmdb_id) or pd.isna(season_number):
        return None

    try:
        tmdb_id = int(tmdb_id)
        season_number = int(season_number)
    except (ValueError, TypeError):
        return None

    url = f'{TMDB_BASE_URL}/tv/{tmdb_id}/season/{season_number}'

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'air_date': data.get('air_date'),
                'name': data.get('name'),
                'episode_count': len(data.get('episodes', [])),
                'overview': data.get('overview', '')[:100]
            }
        elif response.status_code == 404:
            # Season doesn't exist in TMDB
            return {'air_date': None, 'status': 'not_found'}
        else:
            return {'air_date': None, 'status': f'error_{response.status_code}'}
    except requests.exceptions.RequestException as e:
        return {'air_date': None, 'status': f'request_error'}

def batch_fetch_seasons(records):
    """
    Fetch season info for multiple records in parallel.
    """
    results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}

        for idx, row in records.iterrows():
            tmdb_id = row.get('tmdb_id')
            season_num = row.get('season_number', 1)

            if pd.notna(tmdb_id):
                future = executor.submit(get_season_info, tmdb_id, season_num)
                futures[future] = idx
                time.sleep(API_DELAY)  # Rate limiting

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                results[idx] = None

    return results

# Test API connection
print('  Testing TMDB API connection...')
test_result = get_season_info(108978, 1)  # Reacher Season 1
if test_result and test_result.get('air_date'):
    print(f'  API test successful: Reacher S1 air_date = {test_result["air_date"]}')
else:
    print(f'  API test result: {test_result}')

# ============================================================================
# SECTION 4: VALIDATE PREMIERE DATES FOR TV SEASONS
# ============================================================================
print('\n[SECTION 4] Validating premiere dates for TV seasons...')

# Add new columns for tracking
df['premiere_date_validated'] = df.get('premiere_date', pd.NaT)
df['premiere_date_source'] = 'original'
df['release_status_validated'] = df.get('status', 'Unknown')
df['is_unreleased'] = False
df['views_blocked'] = False

if len(tv_seasons) > 0 and 'tmdb_id' in df.columns:
    # Get unique tmdb_id + season_number combinations that need lookup
    tv_with_tmdb = tv_seasons[tv_seasons['tmdb_id'].notna()].copy()

    # Get unique combinations
    if 'season_number' in tv_with_tmdb.columns:
        unique_lookups = tv_with_tmdb[['tmdb_id', 'season_number']].drop_duplicates()
    else:
        unique_lookups = tv_with_tmdb[['tmdb_id']].drop_duplicates()
        unique_lookups['season_number'] = 1

    print(f'  Unique TMDB lookups needed: {len(unique_lookups):,}')

    # Priority titles that MUST be processed first (known issues)
    PRIORITY_TMDB_IDS = [108978]  # Reacher - known Season 4 issue

    # Process priority titles first
    priority_mask = unique_lookups['tmdb_id'].isin(PRIORITY_TMDB_IDS)
    priority_lookups = unique_lookups[priority_mask]
    other_lookups = unique_lookups[~priority_mask]

    # Combine with priority first
    unique_lookups = pd.concat([priority_lookups, other_lookups])

    # For full production run, set to None or very high number
    MAX_LOOKUPS = 100  # None = process all, 100 for quick test with priority titles
    if MAX_LOOKUPS and len(unique_lookups) > MAX_LOOKUPS:
        print(f'  Limiting to {MAX_LOOKUPS} lookups for this run')
        unique_lookups = unique_lookups.head(MAX_LOOKUPS)

    print(f'  Priority titles: {len(priority_lookups)}')

    # Create lookup cache
    lookup_cache = {}

    print(f'  Fetching season data from TMDB API...')
    lookup_start = time.time()

    for i, (idx, row) in enumerate(unique_lookups.iterrows()):
        tmdb_id = int(row['tmdb_id'])
        season_num = int(row.get('season_number', 1))
        cache_key = (tmdb_id, season_num)

        if cache_key not in lookup_cache:
            STATS['tmdb_lookups_attempted'] += 1
            result = get_season_info(tmdb_id, season_num)
            lookup_cache[cache_key] = result

            if result and result.get('air_date'):
                STATS['tmdb_lookups_success'] += 1
            else:
                STATS['tmdb_lookups_failed'] += 1

            time.sleep(API_DELAY)

        # Progress update every 50 lookups
        if (i + 1) % 50 == 0:
            elapsed = time.time() - lookup_start
            rate = (i + 1) / elapsed
            print(f'    Progress: {i+1}/{len(unique_lookups)} lookups ({rate:.1f}/sec)')

    lookup_time = time.time() - lookup_start
    print(f'  TMDB lookups completed in {fmt(lookup_time)}')
    print(f'    Success: {STATS["tmdb_lookups_success"]:,}')
    print(f'    Failed: {STATS["tmdb_lookups_failed"]:,}')

    # Apply validated dates to dataframe
    print('\n  Applying validated dates...')

    for idx, row in df.iterrows():
        # Only process TV seasons with tmdb_id
        if title_type_col and row.get(title_type_col) not in ['tv_season', 'tv', 'series', 'TV Season', 'TV Series']:
            continue

        tmdb_id = row.get('tmdb_id')
        season_num = row.get('season_number', 1)

        if pd.isna(tmdb_id):
            continue

        cache_key = (int(tmdb_id), int(season_num) if pd.notna(season_num) else 1)

        if cache_key in lookup_cache:
            result = lookup_cache[cache_key]

            if result and result.get('air_date'):
                air_date = result['air_date']
                df.at[idx, 'premiere_date_validated'] = air_date
                df.at[idx, 'premiere_date_source'] = 'tmdb_api'
                STATS['dates_updated'] += 1

                # Check if unreleased
                try:
                    air_date_obj = datetime.strptime(air_date, '%Y-%m-%d').date()
                    if air_date_obj > TODAY:
                        df.at[idx, 'is_unreleased'] = True
                        df.at[idx, 'release_status_validated'] = 'Unreleased'
                        STATS['unreleased_flagged'] += 1
                except:
                    pass

            elif result and result.get('status') == 'not_found':
                # Season doesn't exist in TMDB = unreleased
                df.at[idx, 'is_unreleased'] = True
                df.at[idx, 'release_status_validated'] = 'Not Found (Unreleased)'
                STATS['unreleased_flagged'] += 1

# ============================================================================
# SECTION 5: CHECK EXISTING PREMIERE DATES
# ============================================================================
print('\n[SECTION 5] Checking existing premiere dates...')

# For records not updated by TMDB, check existing premiere_date
if 'premiere_date' in df.columns:
    for idx, row in df.iterrows():
        # Skip if already marked as unreleased
        if row['is_unreleased']:
            continue

        # Skip if date was validated by TMDB
        if row['premiere_date_source'] == 'tmdb_api':
            continue

        premiere = row.get('premiere_date')

        if pd.notna(premiere):
            try:
                if isinstance(premiere, str):
                    premiere_date = datetime.strptime(premiere[:10], '%Y-%m-%d').date()
                else:
                    premiere_date = premiere.date() if hasattr(premiere, 'date') else premiere

                if premiere_date > TODAY:
                    df.at[idx, 'is_unreleased'] = True
                    df.at[idx, 'release_status_validated'] = 'Future Premiere'
                    STATS['unreleased_flagged'] += 1
            except:
                pass

print(f'  Total unreleased flagged: {STATS["unreleased_flagged"]:,}')

# ============================================================================
# SECTION 6: BLOCK VIEWS FOR UNRELEASED CONTENT
# ============================================================================
print('\n[SECTION 6] Blocking views for unreleased content...')

views_col = None
for col in ['views_computed', 'views_y', 'views', 'predicted_views']:
    if col in df.columns:
        views_col = col
        break

if views_col:
    print(f'  Using views column: {views_col}')

    # Store original views for unreleased content
    df['views_original'] = df[views_col]

    # Block views for unreleased content
    unreleased_mask = df['is_unreleased'] == True

    if unreleased_mask.sum() > 0:
        blocked_views = df.loc[unreleased_mask, views_col].sum()
        STATS['views_blocked'] = unreleased_mask.sum()
        STATS['views_blocked_total'] = int(blocked_views)

        # Set views to 0 for unreleased content
        df.loc[unreleased_mask, views_col] = 0
        df.loc[unreleased_mask, 'views_blocked'] = True

        print(f'  Blocked {STATS["views_blocked"]:,} records')
        print(f'  Total views blocked: {STATS["views_blocked_total"]:,}')
else:
    print('  No views column found to block')

# ============================================================================
# SECTION 7: SAMPLE OUTPUT - REACHER VALIDATION
# ============================================================================
print('\n[SECTION 7] Validation sample - Reacher...')

# Find Reacher records
if 'primary_title' in df.columns:
    title_col = 'primary_title'
elif 'title' in df.columns:
    title_col = 'title'
elif 'original_title' in df.columns:
    title_col = 'original_title'
else:
    title_col = None

if title_col:
    reacher_mask = df[title_col].str.contains('Reacher', case=False, na=False)
    reacher = df[reacher_mask].copy()

    if len(reacher) > 0:
        print(f'\n  Reacher records found: {len(reacher)}')

        display_cols = [title_col]
        if 'season_number' in df.columns:
            display_cols.append('season_number')
        display_cols.extend(['premiere_date_validated', 'premiere_date_source',
                           'is_unreleased', 'release_status_validated'])
        if views_col:
            display_cols.extend(['views_original', views_col, 'views_blocked'])

        # Filter to existing columns
        display_cols = [c for c in display_cols if c in reacher.columns]

        print('\n  Reacher Validation Results:')
        print('  ' + '-'*80)
        for _, row in reacher[display_cols].head(10).iterrows():
            for col in display_cols:
                print(f'    {col}: {row[col]}')
            print('  ' + '-'*40)
    else:
        print('  No Reacher records found')
else:
    print('  No title column found')

# ============================================================================
# SECTION 8: SAVE OUTPUT
# ============================================================================
print('\n[SECTION 8] Saving output...')

# Generate output filename
output_file = f'{OUTPUT_DIR}/Cranberry_BFD_RELEASE_VALIDATED_{RUN_DATE}_V{VERSION}.parquet'

df.to_parquet(output_file, compression='snappy', index=False)
print(f'  Saved: {output_file}')
print(f'  Size: {os.path.getsize(output_file)/1024/1024:.1f} MB')

# Save validation log
log_file = f'{OUTPUT_DIR}/Components/RELEASE_VALIDATION_LOG_{RUN_DATE}.json'

validation_log = {
    'run_timestamp': datetime.now().isoformat(),
    'run_date': RUN_DATE,
    'version': VERSION,
    'input_file': input_file,
    'output_file': output_file,
    'today_date': TODAY_STR,
    'stats': STATS,
    'lookup_cache_sample': {str(k): v for k, v in list(lookup_cache.items())[:5]} if 'lookup_cache' in dir() else {}
}

os.makedirs(os.path.dirname(log_file), exist_ok=True)
with open(log_file, 'w') as f:
    json.dump(validation_log, f, indent=2, default=str)
print(f'  Saved log: {log_file}')

# ============================================================================
# PROOF COUNTERS
# ============================================================================
total_time = time.time() - start_time

print('\n' + '='*70)
print('ENGINE 4 PROOF COUNTERS')
print('='*70)
print(f'  Total records: {STATS["total_records"]:,}')
print(f'  TV seasons: {STATS["tv_seasons"]:,}')
print(f'  Films: {STATS["films"]:,}')
print(f'  TMDB lookups attempted: {STATS["tmdb_lookups_attempted"]:,}')
print(f'  TMDB lookups success: {STATS["tmdb_lookups_success"]:,}')
print(f'  TMDB lookups failed: {STATS["tmdb_lookups_failed"]:,}')
print(f'  Dates updated: {STATS["dates_updated"]:,}')
print(f'  Unreleased flagged: {STATS["unreleased_flagged"]:,}')
print(f'  Views blocked (records): {STATS["views_blocked"]:,}')
print(f'  Views blocked (total): {STATS["views_blocked_total"]:,}')
print(f'  Total time: {fmt(total_time)}')

print('\n' + '='*70)
print('ENGINE 4 COMPLETE')
print('='*70)
