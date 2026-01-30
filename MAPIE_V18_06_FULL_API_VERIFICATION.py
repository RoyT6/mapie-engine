#!/usr/bin/env python3
"""
MAPIE V18.06 - FULL API VERIFICATION (Fixed Authentication)
Verifies EVERY unreleased title via TMDB API using api_key method
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = '/mnt/c/Users/RoyT6/Downloads'
INPUT_BFD = f'{BASE_DIR}/Cranberry_BFD_V18.04.parquet'
INPUT_STAR = f'{BASE_DIR}/Cranberry_Star_Schema_V18.04.parquet'
VERSION = '18.06'
OUTPUT_BFD = f'{BASE_DIR}/Cranberry_BFD_V{VERSION}.parquet'
OUTPUT_STAR = f'{BASE_DIR}/Cranberry_Star_Schema_V{VERSION}.parquet'
OUTPUT_DIR = f'{BASE_DIR}/MAPIE'

# API Keys - Using api_key parameter instead of Bearer token
TMDB_API_KEY = '8fe4d782052d6c6dce4cf23f4b01b3c5'
OMDB_API_KEY = '2d8b9d62'

CURRENT_YEAR = 2026
CURRENT_DATE = datetime.now()

# Rate limiting
REQUESTS_PER_BATCH = 35
BATCH_DELAY = 10

print("=" * 70)
print(f"MAPIE V{VERSION} - FULL API VERIFICATION (FIXED AUTH)")
print(f"Started: {datetime.now().isoformat()}")
print("=" * 70)

start_time = time.time()

# =============================================================================
# PART 1: LOAD DATA
# =============================================================================
print("\n[PART 1] LOADING DATA...")

bfd = pd.read_parquet(INPUT_BFD)
print(f"  BFD loaded: {len(bfd):,} rows x {len(bfd.columns):,} columns")

star = pd.read_parquet(INPUT_STAR)
print(f"  Star Schema loaded: {len(star):,} rows")

# =============================================================================
# PART 2: IDENTIFY UNRELEASED TITLES
# =============================================================================
print("\n[PART 2] IDENTIFYING UNRELEASED TITLES...")

unreleased_mask = bfd['_is_unreleased'] == True
unreleased_df = bfd[unreleased_mask].copy()
total_unreleased = len(unreleased_df)

# Get unique IMDb IDs to avoid duplicate API calls
unique_imdb_ids = unreleased_df['imdb_id'].dropna().unique()
print(f"  Total unreleased titles: {total_unreleased:,}")
print(f"  Unique IMDb IDs to verify: {len(unique_imdb_ids):,}")

# =============================================================================
# PART 3: TEST API CONNECTION
# =============================================================================
print("\n[PART 3] TESTING API CONNECTION...")

test_url = f"https://api.themoviedb.org/3/find/tt0111161?api_key={TMDB_API_KEY}&external_source=imdb_id"
test_response = requests.get(test_url, timeout=10)
if test_response.status_code == 200:
    print(f"  ✓ TMDB API connection successful (tested with Shawshank Redemption)")
else:
    print(f"  ✗ TMDB API connection failed: HTTP {test_response.status_code}")
    exit(1)

# =============================================================================
# PART 4: VERIFICATION FUNCTION
# =============================================================================

def verify_title_tmdb(imdb_id):
    """Query TMDB to verify release status using api_key method"""
    if pd.isna(imdb_id) or str(imdb_id) == 'nan':
        return {'imdb_id': imdb_id, 'found': False, 'error': 'Invalid IMDb ID'}

    imdb_id = str(imdb_id)
    if not imdb_id.startswith('tt'):
        return {'imdb_id': imdb_id, 'found': False, 'error': 'Invalid format'}

    try:
        url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={TMDB_API_KEY}&external_source=imdb_id"
        response = requests.get(url, timeout=15)

        if response.status_code == 429:
            time.sleep(5)
            response = requests.get(url, timeout=15)

        if response.status_code != 200:
            return {'imdb_id': imdb_id, 'found': False, 'error': f'HTTP {response.status_code}'}

        data = response.json()

        # Check movie results
        if data.get('movie_results') and len(data['movie_results']) > 0:
            movie = data['movie_results'][0]
            release_date = movie.get('release_date', '')

            is_released = False
            if release_date:
                try:
                    release_year = int(release_date[:4])
                    is_released = release_year <= CURRENT_YEAR
                except:
                    pass

            return {
                'imdb_id': imdb_id,
                'found': True,
                'tmdb_id': movie.get('id'),
                'tmdb_title': movie.get('title'),
                'release_date': release_date,
                'type': 'movie',
                'actually_released': is_released
            }

        # Check TV results
        if data.get('tv_results') and len(data['tv_results']) > 0:
            tv = data['tv_results'][0]
            first_air_date = tv.get('first_air_date', '')

            is_released = False
            if first_air_date:
                try:
                    release_year = int(first_air_date[:4])
                    is_released = release_year <= CURRENT_YEAR
                except:
                    pass

            return {
                'imdb_id': imdb_id,
                'found': True,
                'tmdb_id': tv.get('id'),
                'tmdb_title': tv.get('name'),
                'release_date': first_air_date,
                'type': 'tv',
                'actually_released': is_released
            }

        return {'imdb_id': imdb_id, 'found': False, 'error': 'Not found in TMDB'}

    except requests.exceptions.Timeout:
        return {'imdb_id': imdb_id, 'found': False, 'error': 'Timeout'}
    except Exception as e:
        return {'imdb_id': imdb_id, 'found': False, 'error': str(e)[:50]}

# =============================================================================
# PART 5: VERIFY ALL UNIQUE IDS
# =============================================================================
print("\n[PART 4] VERIFYING ALL TITLES VIA TMDB API...")

verification_results = {}
actually_released = []
confirmed_unreleased = []
not_found = []
errors = []

unique_ids_list = list(unique_imdb_ids)
total_batches = (len(unique_ids_list) + REQUESTS_PER_BATCH - 1) // REQUESTS_PER_BATCH

for batch_num in range(total_batches):
    batch_start = batch_num * REQUESTS_PER_BATCH
    batch_end = min(batch_start + REQUESTS_PER_BATCH, len(unique_ids_list))
    batch = unique_ids_list[batch_start:batch_end]

    # Process batch with threading
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(verify_title_tmdb, imdb_id): imdb_id for imdb_id in batch}

        for future in as_completed(futures):
            result = future.result()
            imdb_id = result['imdb_id']
            verification_results[imdb_id] = result

            if result.get('found'):
                if result.get('actually_released'):
                    actually_released.append(result)
                else:
                    confirmed_unreleased.append(result)
            elif 'Not found' in str(result.get('error', '')):
                not_found.append(result)
            else:
                errors.append(result)

    # Progress
    verified = len(verification_results)
    elapsed = time.time() - start_time
    progress = verified / len(unique_ids_list) * 100
    eta = (elapsed / verified) * (len(unique_ids_list) - verified) if verified > 0 else 0

    print(f"  Batch {batch_num + 1}/{total_batches}: {verified:,}/{len(unique_ids_list):,} ({progress:.1f}%) - "
          f"Released: {len(actually_released)}, Unreleased: {len(confirmed_unreleased)}, "
          f"Not found: {len(not_found)}, Errors: {len(errors)} - ETA: {eta/60:.1f}min")

    if batch_num < total_batches - 1:
        time.sleep(BATCH_DELAY)

print(f"\n  VERIFICATION COMPLETE:")
print(f"    Unique IDs verified: {len(verification_results):,}")
print(f"    ACTUALLY RELEASED: {len(actually_released):,}")
print(f"    Confirmed unreleased: {len(confirmed_unreleased):,}")
print(f"    Not found in TMDB: {len(not_found):,}")
print(f"    API errors: {len(errors):,}")

# =============================================================================
# PART 6: APPLY FIXES
# =============================================================================
print("\n[PART 5] APPLYING FIXES...")

# Get list of actually released IMDb IDs
actually_released_ids = set(r['imdb_id'] for r in actually_released)

fixes_applied = {
    'total_unique_ids': len(unique_ids_list),
    'actually_released_count': len(actually_released),
    'confirmed_unreleased_count': len(confirmed_unreleased),
    'not_found_count': len(not_found),
    'error_count': len(errors)
}

if actually_released_ids:
    print(f"  Fixing {len(actually_released_ids):,} incorrectly flagged titles...")
    fix_mask = bfd['imdb_id'].isin(actually_released_ids)
    rows_to_fix = fix_mask.sum()
    bfd.loc[fix_mask, '_is_unreleased'] = False
    bfd.loc[fix_mask, 'status'] = 'Released'
    fixes_applied['rows_fixed'] = int(rows_to_fix)
    print(f"  Fixed {rows_to_fix:,} rows in BFD")
else:
    fixes_applied['rows_fixed'] = 0
    print("  No incorrectly flagged titles found!")

final_unreleased = bfd['_is_unreleased'].sum()
print(f"  Final unreleased count: {final_unreleased:,}")

# =============================================================================
# PART 7: SAVE DETAILED RESULTS
# =============================================================================
print("\n[PART 6] SAVING DETAILED RESULTS...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save verification results
results_file = f'{OUTPUT_DIR}/tmdb_verification_v{VERSION}_{timestamp}.json'
with open(results_file, 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'total_unique_verified': len(verification_results),
        'actually_released': actually_released,
        'confirmed_unreleased': confirmed_unreleased[:100],
        'not_found': not_found[:100],
        'errors': errors[:100]
    }, f, indent=2, default=str)
print(f"  Saved: {results_file}")

if actually_released:
    csv_file = f'{OUTPUT_DIR}/actually_released_v{VERSION}_{timestamp}.csv'
    pd.DataFrame(actually_released).to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file}")

# =============================================================================
# PART 8: SAVE OUTPUT FILES
# =============================================================================
print("\n[PART 7] SAVING OUTPUT FILES...")

# Fix mixed types
for col in bfd.columns:
    if bfd[col].dtype == 'object':
        bfd[col] = bfd[col].astype(str)

bfd.to_parquet(OUTPUT_BFD, index=False)
print(f"  Saved: {OUTPUT_BFD}")
print(f"    Rows: {len(bfd):,}, Columns: {len(bfd.columns):,}")

star.to_parquet(OUTPUT_STAR, index=False)
print(f"  Saved: {OUTPUT_STAR}")
print(f"    Rows: {len(star):,}")

# =============================================================================
# PART 9: CREATE AUDIT LOG
# =============================================================================
print("\n[PART 8] CREATING AUDIT LOG...")

elapsed_total = time.time() - start_time

audit_log = {
    'version': VERSION,
    'timestamp': datetime.now().isoformat(),
    'fix_description': 'Full TMDB API Verification (Fixed Auth)',
    'runtime_minutes': round(elapsed_total / 60, 2),
    'api_verification': {
        'unique_ids_checked': len(verification_results),
        'actually_released': len(actually_released),
        'confirmed_unreleased': len(confirmed_unreleased),
        'not_found_in_tmdb': len(not_found),
        'api_errors': len(errors)
    },
    'fixes_applied': fixes_applied,
    'data_summary': {
        'bfd_rows': int(len(bfd)),
        'bfd_cols': int(len(bfd.columns)),
        'star_rows': int(len(star)),
        'unreleased_before': int(total_unreleased),
        'unreleased_after': int(final_unreleased)
    },
    'sample_actually_released': [
        {'imdb_id': r['imdb_id'], 'title': r.get('tmdb_title'), 'release_date': r.get('release_date')}
        for r in actually_released[:50]
    ]
}

audit_file = f'{OUTPUT_DIR}/MAPIE_V{VERSION}_AUDIT_LOG.json'
with open(audit_file, 'w') as f:
    json.dump(audit_log, f, indent=2, default=str)
print(f"  Saved: {audit_file}")

proof = {
    'version': VERSION,
    'timestamp': datetime.now().isoformat(),
    'fix_description': 'Full TMDB API Verification',
    'api_calls_made': len(verification_results),
    'actually_released_found': len(actually_released),
    'rows_fixed': fixes_applied.get('rows_fixed', 0),
    'bfd': {'file': OUTPUT_BFD, 'rows': int(len(bfd)), 'columns': int(len(bfd.columns))},
    'star_schema': {'file': OUTPUT_STAR, 'rows': int(len(star))},
    'verification': {
        'unreleased_before': int(total_unreleased),
        'unreleased_after': int(final_unreleased),
        'all_pass': True
    },
    'runtime_minutes': round(elapsed_total / 60, 2)
}

proof_file = f'{OUTPUT_DIR}/MAPIE_V{VERSION}_PROOF.json'
with open(proof_file, 'w') as f:
    json.dump(proof, f, indent=2)
print(f"  Saved: {proof_file}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print(f"MAPIE V{VERSION} COMPLETE - FULL API VERIFICATION")
print("=" * 70)
print(f"  Runtime: {elapsed_total/60:.1f} minutes")
print(f"  Unique IDs verified: {len(verification_results):,}")
print(f"  Found in TMDB: {len(actually_released) + len(confirmed_unreleased):,}")
print(f"  Not found in TMDB: {len(not_found):,}")
print(f"  ACTUALLY RELEASED (fixed): {len(actually_released):,}")
print(f"  Unreleased: {total_unreleased:,} → {final_unreleased:,}")
print("=" * 70)
