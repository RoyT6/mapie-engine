#!/usr/bin/env python3
"""
MAPE SCORE CALCULATION V2 - USING NETFLIX SYMPHONY GROUND TRUTH
================================================================
Compares BFD database views against netflix_symphony_combined.parquet
which contains properly parsed Netflix published data with fc_uid.
"""
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('MAPE SCORE CALCULATION V2 - Netflix Symphony Ground Truth')
print('='*80)
print(f'Timestamp: {datetime.now().isoformat()}')
print()

BASE = Path('C:/Users/RoyT6/Downloads')
TRAINING = Path('C:/Users/RoyT6/Downloads/Training Data')

# Load BFD merged database
print('[1] Loading BFD merged database...')
db = pd.read_parquet(BASE / 'BFD-Views-2026-Feb-2.00.parquet')
print(f'    Rows: {len(db):,} | Columns: {len(db.columns)}')

# Normalize fc_uid: strip 'tt' prefix for matching
db['fc_uid_norm'] = db['fc_uid'].apply(lambda x: str(x).replace('tt', '') if pd.notna(x) else None)
print(f'    Sample fc_uid_norm: {db["fc_uid_norm"].dropna().head(3).tolist()}')

# Load Netflix Symphony ground truth
print()
print('[2] Loading Netflix Symphony ground truth...')
nf = pd.read_parquet(TRAINING / 'Qtr-Mths-Countries' / 'netflix_symphony_combined.parquet')
print(f'    Rows: {len(nf):,} | Columns: {len(nf.columns)}')
print(f'    Sample fc_uid: {nf["fc_uid"].dropna().head(3).tolist()}')

# Column mapping: Netflix Symphony -> BFD Database
# Netflix: views_Q1_2023, views_1hy_2023
# BFD: views_q1_2023_total, views_h1_2023_total

# Define periods to compare (using half-year totals as these are the original Netflix published values)
period_mappings = [
    ('views_1hy_2023', 'views_h1_2023_total', 'H1 2023'),
    ('views_2hy_2023', 'views_h2_2023_total', 'H2 2023'),
    ('views_1hy_2024', 'views_h1_2024_total', 'H1 2024'),
    ('views_2hy_2024', 'views_h2_2024_total', 'H2 2024'),
    ('views_1hy_2025', 'views_h1_2025_total', 'H1 2025'),
    ('views_2hy_2025', 'views_h2_2025_total', 'H2 2025'),
]

# Also compare quarterly (more granular)
quarterly_mappings = [
    ('views_Q1_2023', 'views_q1_2023_total', 'Q1 2023'),
    ('views_Q2_2023', 'views_q2_2023_total', 'Q2 2023'),
    ('views_Q3_2023', 'views_q3_2023_total', 'Q3 2023'),
    ('views_Q4_2023', 'views_q4_2023_total', 'Q4 2023'),
    ('views_Q1_2024', 'views_q1_2024_total', 'Q1 2024'),
    ('views_Q2_2024', 'views_q2_2024_total', 'Q2 2024'),
    ('views_Q3_2024', 'views_q3_2024_total', 'Q3 2024'),
    ('views_Q4_2024', 'views_q4_2024_total', 'Q4 2024'),
    ('views_Q1_2025', 'views_q1_2025_total', 'Q1 2025'),
    ('views_Q2_2025', 'views_q2_2025_total', 'Q2 2025'),
    ('views_Q3_2025', 'views_q3_2025_total', 'Q3 2025'),
    ('views_Q4_2025', 'views_q4_2025_total', 'Q4 2025'),
]

# Check which columns exist
print()
print('[3] Checking column availability...')

available_hy = []
for nf_col, db_col, period in period_mappings:
    nf_has = nf_col in nf.columns
    db_has = db_col in db.columns
    if nf_has and db_has:
        available_hy.append((nf_col, db_col, period))
    print(f'    {period}: NF={nf_col}({nf_has}) -> DB={db_col}({db_has})')

available_q = []
for nf_col, db_col, period in quarterly_mappings:
    nf_has = nf_col in nf.columns
    db_has = db_col in db.columns
    if nf_has and db_has:
        available_q.append((nf_col, db_col, period))

print(f'    Available half-year periods: {len(available_hy)}')
print(f'    Available quarterly periods: {len(available_q)}')

# Merge on normalized fc_uid
print()
print('[4] Matching by fc_uid...')

# Keep only needed columns from each
nf_cols = ['fc_uid', 'title', 'title_type'] + [m[0] for m in available_hy] + [m[0] for m in available_q]
nf_cols = [c for c in nf_cols if c in nf.columns]

db_cols = ['fc_uid', 'fc_uid_norm', 'title'] + [m[1] for m in available_hy] + [m[1] for m in available_q]
db_cols = [c for c in db_cols if c in db.columns]

nf_subset = nf[nf_cols].copy()
db_subset = db[db_cols].copy()

# Merge on fc_uid = fc_uid_norm (Netflix fc_uid matches BFD fc_uid without 'tt')
merged = nf_subset.merge(
    db_subset,
    left_on='fc_uid',
    right_on='fc_uid_norm',
    how='inner',
    suffixes=('_nf', '_db')
)

print(f'    Netflix titles: {len(nf_subset):,}')
print(f'    BFD titles: {len(db_subset):,}')
print(f'    Matched titles: {len(merged):,}')

if len(merged) == 0:
    print()
    print('ERROR: No matches found. Checking fc_uid formats...')
    print(f'Netflix fc_uid samples: {nf["fc_uid"].dropna().head(5).tolist()}')
    print(f'BFD fc_uid_norm samples: {db["fc_uid_norm"].dropna().head(5).tolist()}')

    # Try alternative matching
    print()
    print('Trying title-based matching as fallback...')
    nf_subset['title_clean'] = nf_subset['title'].str.lower().str.strip()
    db_subset['title_clean'] = db_subset['title'].str.lower().str.strip()

    merged = nf_subset.merge(
        db_subset,
        on='title_clean',
        how='inner',
        suffixes=('_nf', '_db')
    )
    print(f'    Title-based matches: {len(merged):,}')

# Calculate MAPE for each period
print()
print('[5] Calculating MAPE by period...')

all_comparisons = []

# Half-year comparisons (primary - these are the original Netflix published values)
for nf_col, db_col, period in available_hy:
    if nf_col in merged.columns and db_col in merged.columns:
        valid = merged[(merged[nf_col] > 0) & (merged[db_col] > 0)].copy()
        if len(valid) > 0:
            valid['ape'] = np.abs(valid[db_col] - valid[nf_col]) / valid[nf_col]
            valid['period'] = period
            valid['period_type'] = 'half-year'
            valid['nf_views'] = valid[nf_col]
            valid['db_views'] = valid[db_col]
            all_comparisons.append(valid[['title_nf' if 'title_nf' in valid.columns else 'title',
                                          'fc_uid_nf' if 'fc_uid_nf' in valid.columns else 'fc_uid',
                                          'nf_views', 'db_views', 'ape', 'period', 'period_type']])
            mape = valid['ape'].mean() * 100
            print(f'    {period}: MAPE={mape:.2f}% ({len(valid):,} titles)')

# Quarterly comparisons
for nf_col, db_col, period in available_q:
    if nf_col in merged.columns and db_col in merged.columns:
        valid = merged[(merged[nf_col] > 0) & (merged[db_col] > 0)].copy()
        if len(valid) > 0:
            valid['ape'] = np.abs(valid[db_col] - valid[nf_col]) / valid[nf_col]
            valid['period'] = period
            valid['period_type'] = 'quarterly'
            valid['nf_views'] = valid[nf_col]
            valid['db_views'] = valid[db_col]
            all_comparisons.append(valid[['title_nf' if 'title_nf' in valid.columns else 'title',
                                          'fc_uid_nf' if 'fc_uid_nf' in valid.columns else 'fc_uid',
                                          'nf_views', 'db_views', 'ape', 'period', 'period_type']])
            mape = valid['ape'].mean() * 100
            print(f'    {period}: MAPE={mape:.2f}% ({len(valid):,} titles)')

# Combine all comparisons
if all_comparisons:
    all_data = pd.concat(all_comparisons, ignore_index=True)

    # Rename columns for consistency
    if 'title_nf' in all_data.columns:
        all_data = all_data.rename(columns={'title_nf': 'title', 'fc_uid_nf': 'fc_uid'})

    # Filter valid APE (< 1000%)
    valid = all_data[all_data['ape'] < 10]

    print()
    print('='*80)
    print('MAPE RESULTS - NETFLIX SYMPHONY VS BFD DATABASE')
    print('='*80)
    print(f'  Total Comparisons:  {len(all_data):,}')
    print(f'  Valid Comparisons:  {len(valid):,}')

    mape = valid['ape'].mean() * 100
    median_ape = valid['ape'].median() * 100

    print()
    print(f'  MAPE:               {mape:.2f}%')
    print(f'  Median APE:         {median_ape:.2f}%')

    # Correlation
    corr = valid['db_views'].corr(valid['nf_views'])
    print(f'  Correlation (r):    {corr:.4f}')
    print(f'  R-squared:          {corr**2:.4f}')

    # By period type
    print()
    print('  MAPE by Period Type:')
    for pt in valid['period_type'].unique():
        pt_data = valid[valid['period_type'] == pt]
        pt_mape = pt_data['ape'].mean() * 100
        pt_corr = pt_data['db_views'].corr(pt_data['nf_views'])
        print(f'    {pt}: MAPE={pt_mape:.2f}% r={pt_corr:.4f} ({len(pt_data):,} comparisons)')

    # By period
    print()
    print('  MAPE by Period:')
    for period in sorted(valid['period'].unique()):
        period_data = valid[valid['period'] == period]
        period_mape = period_data['ape'].mean() * 100
        print(f'    {period}: {period_mape:.2f}% ({len(period_data):,} titles)')

    # Error distribution
    print()
    print('  Error Distribution:')
    ranges = [(0, 1), (1, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 500)]
    for low, high in ranges:
        count = ((valid['ape']*100 >= low) & (valid['ape']*100 < high)).sum()
        pct = count / len(valid) * 100
        print(f'    {low:>4}% - {high:<4}%: {count:>6,} titles ({pct:>5.1f}%)')

    # Best matches
    print()
    print('  Top 15 Best Matches (Lowest APE):')
    top = valid.nsmallest(15, 'ape')
    for _, row in top.iterrows():
        title = str(row.get('title', 'Unknown'))[:40].encode('ascii', 'replace').decode()
        print(f'    {title:<40} NF:{row["nf_views"]:>12,.0f} DB:{row["db_views"]:>12,.0f} APE:{row["ape"]*100:>6.2f}%')

    # Worst matches
    print()
    print('  Top 15 Worst Matches (Highest APE, APE < 1000%):')
    worst = valid.nlargest(15, 'ape')
    for _, row in worst.iterrows():
        title = str(row.get('title', 'Unknown'))[:40].encode('ascii', 'replace').decode()
        print(f'    {title:<40} NF:{row["nf_views"]:>12,.0f} DB:{row["db_views"]:>12,.0f} APE:{row["ape"]*100:>6.2f}%')

    # Anti-cheat validation
    print()
    print('='*80)
    print('ANTI-CHEAT VALIDATION')
    print('='*80)
    print(f'  MAPE Valid Range:   5% - 40%')
    print(f'  Actual MAPE:        {mape:.2f}%')
    status = 'PASS' if 5.0 <= mape <= 40.0 else ('REVIEW - TOO LOW' if mape < 5.0 else 'HIGH')
    print(f'  Status:             {status}')

    print(f'  R² Valid Range:     0.30 - 0.90')
    print(f'  Actual R²:          {corr**2:.4f}')
    r2_status = 'PASS' if 0.30 <= corr**2 <= 0.90 else ('REVIEW - TOO HIGH' if corr**2 > 0.90 else 'LOW')
    print(f'  R² Status:          {r2_status}')
    print('='*80)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'methodology': 'netflix_symphony_combined.parquet vs BFD database by fc_uid',
        'ground_truth_file': 'netflix_symphony_combined.parquet',
        'database_file': 'BFD-Views-2026-Feb-2.00.parquet',
        'total_comparisons': int(len(all_data)),
        'valid_comparisons': int(len(valid)),
        'matched_titles': int(len(merged)),
        'mape_percent': round(mape, 2),
        'median_ape_percent': round(median_ape, 2),
        'correlation': round(corr, 4),
        'r_squared': round(corr**2, 4),
        'by_period_type': {pt: {'mape': round(valid[valid['period_type']==pt]['ape'].mean()*100, 2),
                                'count': int(len(valid[valid['period_type']==pt]))}
                         for pt in valid['period_type'].unique()},
        'by_period': {p: {'mape': round(valid[valid['period']==p]['ape'].mean()*100, 2),
                         'count': int(len(valid[valid['period']==p]))}
                     for p in valid['period'].unique()},
        'anti_cheat_check': {
            'mape_in_valid_range': bool(5.0 <= mape <= 40.0),
            'mape_valid_range': '5% - 40%',
            'mape_status': status,
            'r2_in_valid_range': bool(0.30 <= corr**2 <= 0.90),
            'r2_valid_range': '0.30 - 0.90',
            'r2_status': r2_status
        }
    }

    output_file = BASE / 'MAPIE Engine' / f'MAPE_SCORES_V2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved: {output_file}')
else:
    print('\nERROR: No comparisons generated')
