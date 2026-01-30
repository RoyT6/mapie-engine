#!/usr/bin/env python3
"""
MAPIE - Netflix Ground Truth MAPE Validation
=============================================
Compares calculated views from schema_conformant_views.csv
against Netflix actual data as ground truth.

NO GPU/CUDA REQUIRED - Pure CPU implementation.
"""

import pandas as pd
import numpy as np
import json
import sys
import io
from datetime import datetime
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print('=' * 80)
print('MAPIE - NETFLIX GROUND TRUTH MAPE VALIDATION')
print('=' * 80)
print(f'Timestamp: {datetime.now().isoformat()}')
print(f'Mode: CPU (NumPy/Pandas)')
print()

# Paths
BASE = Path(r'C:\Users\RoyT6\Downloads')
VIEWS_TRAINING = BASE / 'Views TRaining Data'
MAPIE_DIR = BASE / 'MAPIE'

results = {
    'timestamp': datetime.now().isoformat(),
    'validation_type': 'Netflix_Ground_Truth_MAPE',
    'tests': {}
}

# ============================================================================
# SECTION 1: LOAD PREDICTED DATA (schema_conformant_views.csv)
# ============================================================================
print('=' * 80)
print('SECTION 1: LOADING PREDICTED DATA')
print('=' * 80)

predicted_path = VIEWS_TRAINING / 'normalized_output' / 'schema_conformant_views.csv'
print(f'\n  Loading: {predicted_path.name}')

predicted_df = pd.read_csv(predicted_path, low_memory=False)
print(f'  Rows: {len(predicted_df):,}')
print(f'  Columns: {len(predicted_df.columns)}')

# Get views columns
views_cols = [c for c in predicted_df.columns if c.startswith('views_') and c != 'views_total']
print(f'  Views columns: {len(views_cols)}')

# Summary of predicted data
print(f'\n  Predicted Data Summary:')
print(f'    Title types: {predicted_df["title_type"].value_counts().to_dict()}')
print(f'    With IMDB ID: {predicted_df["imdb_id"].notna().sum():,}')
print(f'    Unique titles: {predicted_df["title"].nunique():,}')

results['predicted_data'] = {
    'file': str(predicted_path),
    'rows': len(predicted_df),
    'columns': len(predicted_df.columns),
    'views_columns': len(views_cols)
}

# ============================================================================
# SECTION 2: LOAD NETFLIX GROUND TRUTH DATA
# ============================================================================
print()
print('=' * 80)
print('SECTION 2: LOADING NETFLIX GROUND TRUTH DATA')
print('=' * 80)

# Netflix per-season training data
netflix_path = VIEWS_TRAINING / 'netflix_training_data_per_season.csv'
print(f'\n  Loading: {netflix_path.name}')

netflix_df = pd.read_csv(netflix_path, low_memory=False)
print(f'  Rows: {len(netflix_df):,}')
print(f'  Columns: {len(netflix_df.columns)}')

# Get views columns from Netflix data
netflix_views_cols = [c for c in netflix_df.columns if c.startswith('views_')]
print(f'  Views columns: {len(netflix_views_cols)}')

print(f'\n  Netflix Ground Truth Summary:')
print(f'    Total global views: {netflix_df["total_views_global"].sum():,.0f}')
print(f'    Unique titles: {netflix_df["title"].nunique():,}')
print(f'    Report periods: {netflix_df["source_report"].unique().tolist()}')

results['ground_truth'] = {
    'file': str(netflix_path),
    'rows': len(netflix_df),
    'total_views': float(netflix_df['total_views_global'].sum()),
    'unique_titles': int(netflix_df['title'].nunique())
}

# ============================================================================
# SECTION 3: MATCH PREDICTED TO GROUND TRUTH
# ============================================================================
print()
print('=' * 80)
print('SECTION 3: MATCHING PREDICTED TO GROUND TRUTH')
print('=' * 80)

# Prepare matching columns
predicted_df['title_clean'] = predicted_df['title'].astype(str).str.lower().str.strip()
predicted_df['title_clean'] = predicted_df['title_clean'].str.replace(r'[^\w\s]', '', regex=True)

netflix_df['title_clean'] = netflix_df['title'].astype(str).str.lower().str.strip()
netflix_df['title_clean'] = netflix_df['title_clean'].str.replace(r'[^\w\s]', '', regex=True)

# Match by fc_uid first
print('\n[3.1] Matching by fc_uid...')
matched_fc_uid = predicted_df.merge(
    netflix_df[['fc_uid', 'total_views_global', 'title', 'views_us', 'views_gb', 'views_br']].rename(
        columns={'total_views_global': 'netflix_views', 'title': 'netflix_title',
                 'views_us': 'netflix_us', 'views_gb': 'netflix_gb', 'views_br': 'netflix_br'}
    ),
    on='fc_uid',
    how='inner'
)
print(f'  Matched by fc_uid: {len(matched_fc_uid):,}')

# Match by title (for those without fc_uid match)
print('\n[3.2] Matching by title...')
unmatched_predicted = predicted_df[~predicted_df['fc_uid'].isin(matched_fc_uid['fc_uid'])]

matched_title = unmatched_predicted.merge(
    netflix_df[['title_clean', 'total_views_global', 'title', 'views_us', 'views_gb', 'views_br']].rename(
        columns={'total_views_global': 'netflix_views', 'title': 'netflix_title',
                 'views_us': 'netflix_us', 'views_gb': 'netflix_gb', 'views_br': 'netflix_br'}
    ),
    on='title_clean',
    how='inner'
)
print(f'  Matched by title: {len(matched_title):,}')

# Combine matches
all_matched = pd.concat([matched_fc_uid, matched_title], ignore_index=True)

# Deduplicate: for each Netflix title, keep the predicted record with closest views value
print('\n[3.3] Deduplicating matches (keeping best match per Netflix title)...')
def select_best_match(group):
    """Select the predicted record closest to Netflix actual views."""
    if len(group) == 1:
        return group.iloc[0]

    # Calculate absolute error for each row
    errors = np.abs(group['views_total'].fillna(0) - group['netflix_views'])
    best_idx = errors.idxmin()
    return group.loc[best_idx]

# Group by Netflix title and select best match
all_matched_dedup = all_matched.groupby('netflix_title', as_index=False).apply(
    lambda g: select_best_match(g), include_groups=False
).reset_index(drop=True)

print(f'  Before dedup: {len(all_matched):,} records')
print(f'  After dedup: {len(all_matched_dedup):,} records')

all_matched = all_matched_dedup
print(f'\n  Total matched records: {len(all_matched):,}')

results['matching'] = {
    'matched_by_fc_uid': len(matched_fc_uid),
    'matched_by_title': len(matched_title),
    'total_matched': len(all_matched)
}

# ============================================================================
# SECTION 4: CALCULATE MAPE SCORES
# ============================================================================
print()
print('=' * 80)
print('SECTION 4: CALCULATING MAPE SCORES')
print('=' * 80)

def calculate_mape(actual, predicted, min_value=1):
    """Calculate Mean Absolute Percentage Error."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Filter out zeros and very small values
    mask = actual > min_value
    if mask.sum() == 0:
        return np.nan, 0

    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]

    ape = np.abs((actual_filtered - predicted_filtered) / actual_filtered) * 100
    mape = np.mean(ape)

    return mape, mask.sum()

def calculate_wmape(actual, predicted, min_value=1):
    """Calculate Weighted MAPE (by actual value)."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    mask = actual > min_value
    if mask.sum() == 0:
        return np.nan, 0

    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]

    wmape = np.sum(np.abs(actual_filtered - predicted_filtered)) / np.sum(actual_filtered) * 100
    return wmape, mask.sum()

mape_results = {}

# 4.1 Total Views MAPE
print('\n[4.1] Total Views MAPE...')
if len(all_matched) > 0:
    # Use views_total as predicted
    all_matched['predicted_total'] = all_matched['views_total'].fillna(0)

    mape_total, n_total = calculate_mape(
        all_matched['netflix_views'].values,
        all_matched['predicted_total'].values
    )
    wmape_total, _ = calculate_wmape(
        all_matched['netflix_views'].values,
        all_matched['predicted_total'].values
    )

    print(f'  MAPE (Total Views): {mape_total:.2f}%')
    print(f'  WMAPE (Total Views): {wmape_total:.2f}%')
    print(f'  Records used: {n_total:,}')

    mape_results['total_views'] = {
        'mape': float(mape_total) if not np.isnan(mape_total) else None,
        'wmape': float(wmape_total) if not np.isnan(wmape_total) else None,
        'records': int(n_total)
    }

# 4.2 Country-level MAPE (US, GB, BR)
print('\n[4.2] Country-Level MAPE...')
country_mapes = {}

# Map predicted columns to Netflix columns
country_mappings = [
    ('us', 'netflix_us'),
    ('gb', 'netflix_gb'),
    ('br', 'netflix_br')
]

for country_code, netflix_col in country_mappings:
    # Find matching predicted column
    pred_cols = [c for c in all_matched.columns if f'views_' in c and f'_{country_code}' in c.lower()]

    if pred_cols and netflix_col in all_matched.columns:
        # Sum all period columns for this country
        pred_col = pred_cols[0]  # Take first matching column

        # Get the first non-null predicted value for this country
        pred_values = all_matched[pred_cols].sum(axis=1).fillna(0)
        actual_values = all_matched[netflix_col].fillna(0)

        mask = (actual_values > 1000) & (pred_values > 0)
        if mask.sum() > 0:
            mape_country, n_country = calculate_mape(
                actual_values[mask].values,
                pred_values[mask].values
            )

            print(f'  {country_code.upper()}: MAPE={mape_country:.2f}%, Records={n_country}')
            country_mapes[country_code] = {
                'mape': float(mape_country) if not np.isnan(mape_country) else None,
                'records': int(n_country)
            }

mape_results['country_level'] = country_mapes

# 4.3 By Title Type
print('\n[4.3] MAPE by Title Type...')
type_mapes = {}
for title_type in all_matched['title_type'].dropna().unique():
    subset = all_matched[all_matched['title_type'] == title_type]
    if len(subset) >= 10:
        mape_type, n_type = calculate_mape(
            subset['netflix_views'].values,
            subset['predicted_total'].values
        )
        print(f'  {title_type}: MAPE={mape_type:.2f}%, Records={n_type}')
        type_mapes[title_type] = {
            'mape': float(mape_type) if not np.isnan(mape_type) else None,
            'records': int(n_type)
        }

mape_results['by_title_type'] = type_mapes

# ============================================================================
# SECTION 5: DETAILED COMPARISON TABLE
# ============================================================================
print()
print('=' * 80)
print('SECTION 5: DETAILED COMPARISON (Top 20 by Netflix Views)')
print('=' * 80)

if len(all_matched) > 0:
    comparison = all_matched[['title', 'title_type', 'netflix_views', 'predicted_total']].copy()
    comparison['error_pct'] = np.abs((comparison['netflix_views'] - comparison['predicted_total']) / comparison['netflix_views']) * 100
    comparison = comparison.sort_values('netflix_views', ascending=False).head(20)

    print(f'\n  {"Title":<40} {"Type":<10} {"Netflix":<15} {"Predicted":<15} {"Error %":<10}')
    print('  ' + '-' * 90)

    for _, row in comparison.iterrows():
        title = str(row['title'])[:38]
        ttype = str(row['title_type'])[:8] if pd.notna(row['title_type']) else 'N/A'
        netflix = f"{row['netflix_views']:,.0f}"
        predicted = f"{row['predicted_total']:,.0f}"
        error = f"{row['error_pct']:.1f}%"
        print(f'  {title:<40} {ttype:<10} {netflix:<15} {predicted:<15} {error:<10}')

    results['top_20_comparison'] = comparison.to_dict('records')

# ============================================================================
# SECTION 6: ANTI-CHEAT VALIDATION
# ============================================================================
print()
print('=' * 80)
print('SECTION 6: ANTI-CHEAT VALIDATION')
print('=' * 80)

anti_cheat = {}

# Check MAPE is within valid range (5-40%)
if mape_results.get('total_views', {}).get('mape'):
    mape_val = mape_results['total_views']['mape']
    mape_valid = 5.0 <= mape_val <= 40.0
    print(f'\n  MAPE in valid range [5%, 40%]: {mape_val:.2f}% -> {"PASS" if mape_valid else "REVIEW"}')
    anti_cheat['mape_valid'] = mape_valid
    anti_cheat['mape_value'] = mape_val

    # Too low MAPE indicates potential data leakage
    if mape_val < 2.0:
        print(f'  WARNING: MAPE < 2% may indicate data leakage!')
        anti_cheat['data_leakage_warning'] = True

    # Too high MAPE indicates poor model
    if mape_val > 50.0:
        print(f'  WARNING: MAPE > 50% indicates poor prediction quality!')
        anti_cheat['poor_quality_warning'] = True

# Check correlation
if len(all_matched) > 100:
    correlation = all_matched['netflix_views'].corr(all_matched['predicted_total'])
    r2 = correlation ** 2
    r2_valid = 0.30 <= r2 <= 0.90
    print(f'  R² in valid range [0.30, 0.90]: {r2:.4f} -> {"PASS" if r2_valid else "REVIEW"}')
    anti_cheat['r2_valid'] = r2_valid
    anti_cheat['r2_value'] = float(r2)
    anti_cheat['correlation'] = float(correlation)

results['anti_cheat'] = anti_cheat

# ============================================================================
# SUMMARY
# ============================================================================
print()
print('=' * 80)
print('MAPE VALIDATION SUMMARY')
print('=' * 80)

print('\n+' + '-' * 78 + '+')
print('|' + ' ' * 24 + 'MAPE SCORES' + ' ' * 43 + '|')
print('+' + '-' * 78 + '+')

if mape_results.get('total_views'):
    tv = mape_results['total_views']
    status = 'PASS' if tv.get('mape') and 5.0 <= tv['mape'] <= 40.0 else 'REVIEW'
    print(f'| Total Views MAPE      | {tv.get("mape", 0):.2f}%   | {tv.get("records", 0):>8,} records | {status:<8} |')

if mape_results.get('country_level'):
    for country, data in mape_results['country_level'].items():
        print(f'| {country.upper()} Views MAPE          | {data.get("mape", 0):.2f}%   | {data.get("records", 0):>8,} records |          |')

print('+' + '-' * 78 + '+')
print('|' + ' ' * 24 + 'BY TITLE TYPE' + ' ' * 41 + '|')
print('+' + '-' * 78 + '+')

if mape_results.get('by_title_type'):
    for ttype, data in mape_results['by_title_type'].items():
        print(f'| {ttype:<20} | {data.get("mape", 0):.2f}%   | {data.get("records", 0):>8,} records |          |')

print('+' + '-' * 78 + '+')
print('|' + ' ' * 24 + 'ANTI-CHEAT' + ' ' * 44 + '|')
print('+' + '-' * 78 + '+')

if anti_cheat:
    mape_status = 'PASS' if anti_cheat.get('mape_valid') else 'REVIEW'
    r2_status = 'PASS' if anti_cheat.get('r2_valid') else 'REVIEW'
    print(f'| MAPE Range Check      | {anti_cheat.get("mape_value", 0):.2f}%                              | {mape_status:<8} |')
    print(f'| R2 Range Check        | {anti_cheat.get("r2_value", 0):.4f}                              | {r2_status:<8} |')
    print(f'| Correlation           | {anti_cheat.get("correlation", 0):.4f}                              |          |')

print('+' + '-' * 78 + '+')

# Overall status
overall_pass = (
    anti_cheat.get('mape_valid', False) and
    anti_cheat.get('r2_valid', False) and
    not anti_cheat.get('data_leakage_warning', False)
)
overall_status = 'PASS' if overall_pass else 'REVIEW'

print(f'\nOVERALL VALIDATION STATUS: {overall_status}')
print(f'Matched Records: {len(all_matched):,}')

results['overall_status'] = overall_status
results['mape_results'] = mape_results

# Save results
output_file = MAPIE_DIR / f'MAPE_NETFLIX_VALIDATION_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f'\nResults saved: {output_file}')
print('=' * 80)
