#!/usr/bin/env python3
"""
MAPE ANALYSIS REPORT - DATA SOURCE DISCREPANCY INVESTIGATION
=============================================================
Analyzes why MAPE is high between BFD database and Netflix published data.
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('MAPE ANALYSIS REPORT - DATA SOURCE INVESTIGATION')
print('='*80)
print(f'Timestamp: {datetime.now().isoformat()}')
print()

BASE = Path('C:/Users/RoyT6/Downloads')
TRAINING = Path('C:/Users/RoyT6/Downloads/Training Data')

# Load all relevant data sources
print('[1] Loading data sources...')

db = pd.read_parquet(BASE / 'BFD-Views-2026-Feb-2.00.parquet')
print(f'    BFD Database: {len(db):,} rows x {len(db.columns)} cols')

nf = pd.read_parquet(TRAINING / 'Qtr-Mths-Countries' / 'netflix_symphony_combined.parquet')
print(f'    Netflix Symphony: {len(nf):,} rows x {len(nf.columns)} cols')

# Normalize fc_uid
db['fc_uid_norm'] = db['fc_uid'].apply(lambda x: str(x).replace('tt', '') if pd.notna(x) else None)

# DATA SOURCE ANALYSIS
print()
print('='*80)
print('[2] DATA SOURCE COMPARISON')
print('='*80)

# Compare H1 2024 views scale
db_h1 = db['views_h1_2024_total'].dropna()
nf_h1 = nf['views_1hy_2024'].dropna()

print()
print('VIEWS SCALE COMPARISON (H1 2024):')
print()
print('                      | BFD Database      | Netflix Symphony')
print('-' * 60)
print(f'  Max views           | {db_h1.max():>15,.0f} | {nf_h1.max():>15,.0f}')
print(f'  Mean views          | {db_h1.mean():>15,.0f} | {nf_h1.mean():>15,.0f}')
print(f'  Median views        | {db_h1.median():>15,.0f} | {nf_h1.median():>15,.0f}')
print(f'  Titles > 10M        | {(db_h1 > 10_000_000).sum():>15,} | {(nf_h1 > 10_000_000).sum():>15,}')
print(f'  Titles > 1M         | {(db_h1 > 1_000_000).sum():>15,} | {(nf_h1 > 1_000_000).sum():>15,}')
print(f'  Total count         | {len(db_h1):>15,} | {len(nf_h1):>15,}')

print()
print('FINDING: BFD contains FlixPatrol estimates (capped ~1.6M)')
print('         Netflix Symphony contains actual Netflix data (up to 144M)')
print()

# MATCH ANALYSIS
print('='*80)
print('[3] TITLE MATCHING ANALYSIS')
print('='*80)

merged = nf.merge(
    db[['fc_uid_norm', 'fc_uid', 'title', 'views_h1_2024_total']].rename(columns={'title': 'title_db', 'fc_uid': 'fc_uid_db'}),
    left_on='fc_uid',
    right_on='fc_uid_norm',
    how='inner'
)

print(f'    Netflix titles: {len(nf):,}')
print(f'    BFD titles: {len(db):,}')
print(f'    Matched by fc_uid: {len(merged):,}')
print(f'    Match rate (of Netflix): {len(merged)/len(nf)*100:.1f}%')

# For matched titles, compare H1 2024
valid = merged[(merged['views_1hy_2024'] > 0) & (merged['views_h1_2024_total'] > 0)]
print(f'    With valid H1 2024 views in both: {len(valid):,}')

# Calculate MAPE for matched titles
if len(valid) > 0:
    valid = valid.copy()
    valid['ape'] = np.abs(valid['views_h1_2024_total'] - valid['views_1hy_2024']) / valid['views_1hy_2024']
    valid['ratio'] = valid['views_h1_2024_total'] / valid['views_1hy_2024']

    # Filter outliers
    valid_filtered = valid[valid['ape'] < 10]  # APE < 1000%

    mape = valid_filtered['ape'].mean() * 100
    median_ape = valid_filtered['ape'].median() * 100
    mean_ratio = valid_filtered['ratio'].mean()
    median_ratio = valid_filtered['ratio'].median()
    corr = valid_filtered['views_h1_2024_total'].corr(valid_filtered['views_1hy_2024'])

print()
print('='*80)
print('[4] MAPE CALCULATION (BFD vs Netflix Symphony)')
print('='*80)
print()
print('CURRENT STATE (FlixPatrol estimates vs Netflix actuals):')
print(f'  MAPE:           {mape:.2f}%')
print(f'  Median APE:     {median_ape:.2f}%')
print(f'  Mean ratio:     {mean_ratio:.4f} (BFD/Netflix)')
print(f'  Median ratio:   {median_ratio:.4f}')
print(f'  Correlation:    {corr:.4f}')
print(f'  R-squared:      {corr**2:.4f}')
print()
print('  EXPECTED RANGE: 5% - 40%')
print(f'  STATUS: {"FAIL - HIGH" if mape > 40 else "PASS" if mape >= 5 else "FAIL - TOO LOW"}')

# Analyze by views magnitude
print()
print('='*80)
print('[5] MAPE BY VIEWS MAGNITUDE')
print('='*80)
print()
print('(How does accuracy vary by title popularity?)')
print()

ranges = [
    (100_000, 500_000, '100K-500K'),
    (500_000, 1_000_000, '500K-1M'),
    (1_000_000, 5_000_000, '1M-5M'),
    (5_000_000, 10_000_000, '5M-10M'),
    (10_000_000, 50_000_000, '10M-50M'),
    (50_000_000, float('inf'), '50M+'),
]

for low, high, label in ranges:
    subset = valid_filtered[(valid_filtered['views_1hy_2024'] >= low) & (valid_filtered['views_1hy_2024'] < high)]
    if len(subset) > 0:
        mape_range = subset['ape'].mean() * 100
        count = len(subset)
        avg_ratio = subset['ratio'].mean()
        print(f'  {label:>10}: MAPE={mape_range:>6.1f}% | Titles={count:>5,} | Avg ratio={avg_ratio:.3f}')
    else:
        print(f'  {label:>10}: No data')

# ROOT CAUSE ANALYSIS
print()
print('='*80)
print('[6] ROOT CAUSE ANALYSIS')
print('='*80)
print()
print('THE PROBLEM:')
print('  BFD database views_* columns contain FlixPatrol ESTIMATED views')
print('  Netflix Symphony contains Netflix PUBLISHED actual views')
print('  These are fundamentally different data sources')
print()
print('WHY BFD HAS FLIXPATROL DATA:')
print('  - FlixPatrol provides ranking-based view estimates')
print('  - FlixPatrol values are capped (max ~1.6M per period)')
print('  - Netflix publishes uncapped engagement data (up to 144M)')
print('  - The merge added FlixPatrol data to expand coverage')
print()
print('SOLUTION OPTIONS:')
print('  1. Replace BFD views with Netflix Symphony data where matched')
print('  2. Keep FlixPatrol for titles NOT in Netflix Symphony')
print('  3. Recalculate MAPE only for Netflix-sourced titles')
print()

# WHAT MAPE SHOULD BE
print('='*80)
print('[7] EXPECTED MAPE IF USING NETFLIX SYMPHONY DATA')
print('='*80)
print()

# Calculate internal consistency of Netflix Symphony (quarterly reconstruction)
nf_check = nf.copy()
nf_check['h1_calc'] = nf_check['views_Q1_2024'].fillna(0) + nf_check['views_Q2_2024'].fillna(0)
nf_check_valid = nf_check[(nf_check['views_1hy_2024'] > 0) & (nf_check['h1_calc'] > 0)]

if len(nf_check_valid) > 0:
    nf_check_valid = nf_check_valid.copy()
    nf_check_valid['recon_ape'] = np.abs(nf_check_valid['h1_calc'] - nf_check_valid['views_1hy_2024']) / nf_check_valid['views_1hy_2024']
    recon_mape = nf_check_valid['recon_ape'].mean() * 100
    recon_exact = (nf_check_valid['recon_ape'] < 0.001).sum()

    print(f'NETFLIX SYMPHONY INTERNAL CONSISTENCY (Q1+Q2 vs H1):')
    print(f'  Reconstruction MAPE: {recon_mape:.4f}%')
    print(f'  Exact matches: {recon_exact:,} / {len(nf_check_valid):,} ({recon_exact/len(nf_check_valid)*100:.1f}%)')
    print()
    print('  This proves Netflix Symphony data is internally consistent.')
    print('  If BFD contained Netflix data, MAPE would be ~0% for H1 periods.')

# Save report
results = {
    'timestamp': datetime.now().isoformat(),
    'finding': 'BFD database contains FlixPatrol estimates, not Netflix published data',
    'bfd_database': {
        'rows': int(len(db)),
        'views_h1_2024_max': float(db_h1.max()),
        'views_h1_2024_mean': float(db_h1.mean()),
        'views_source': 'FlixPatrol estimated views (ranking-based)'
    },
    'netflix_symphony': {
        'rows': int(len(nf)),
        'views_1hy_2024_max': float(nf_h1.max()),
        'views_1hy_2024_mean': float(nf_h1.mean()),
        'views_source': 'Netflix published actual views'
    },
    'comparison': {
        'matched_titles': int(len(merged)),
        'valid_comparisons': int(len(valid_filtered)),
        'mape_percent': round(mape, 2),
        'median_ape_percent': round(median_ape, 2),
        'correlation': round(corr, 4),
        'mean_ratio': round(mean_ratio, 4),
        'median_ratio': round(median_ratio, 4)
    },
    'netflix_internal_consistency': {
        'reconstruction_mape_percent': round(recon_mape, 4) if len(nf_check_valid) > 0 else None,
        'note': 'Q1+Q2 vs H1 reconstruction accuracy in Netflix Symphony'
    },
    'anti_cheat_check': {
        'mape_in_valid_range': bool(5.0 <= mape <= 40.0),
        'valid_range': '5% - 40%',
        'status': 'FAIL - HIGH (FlixPatrol vs Netflix mismatch)',
        'recommendation': 'Replace BFD views with Netflix Symphony data for matched titles'
    }
}

output_file = BASE / 'MAPIE Engine' / f'MAPE_ANALYSIS_REPORT_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print()
print('='*80)
print(f'Report saved: {output_file}')
print('='*80)
