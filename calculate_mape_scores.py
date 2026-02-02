#!/usr/bin/env python3
"""
MAPE SCORE CALCULATION - FIXED
==============================
Properly aligns FlixPatrol quarterly data with Netflix semi-annual published views
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
print('MAPE SCORE CALCULATION - FlixPatrol vs Netflix Actuals')
print('='*80)
print(f'Timestamp: {datetime.now().isoformat()}')
print()

BASE = Path('C:/Users/RoyT6/Downloads')
ORIGINALS = Path('C:/Users/RoyT6/Downloads/Training Data/Originals')

# Load merged database
print('[1] Loading merged database...')
db = pd.read_parquet(BASE / 'BFD-Views-2026-Feb-2.00.parquet')
print(f'    Rows: {len(db):,} | Columns: {len(db.columns)}')

# Create H1/H2 aggregations from quarterly data (Q1+Q2=H1, Q3+Q4=H2)
print()
print('[2] Aggregating quarterly views to semi-annual (H1/H2)...')

for year in ['2024', '2025']:
    # H1 = Q1 + Q2
    q1_col = f'views_q1_{year}_total'
    q2_col = f'views_q2_{year}_total'
    h1_col = f'views_h1_{year}_total'
    if q1_col in db.columns and q2_col in db.columns:
        db[h1_col] = db[q1_col].fillna(0) + db[q2_col].fillna(0)
        print(f'    Created {h1_col} from {q1_col} + {q2_col}')

    # H2 = Q3 + Q4
    q3_col = f'views_q3_{year}_total'
    q4_col = f'views_q4_{year}_total'
    h2_col = f'views_h2_{year}_total'
    if q3_col in db.columns and q4_col in db.columns:
        db[h2_col] = db[q3_col].fillna(0) + db[q4_col].fillna(0)
        print(f'    Created {h2_col} from {q3_col} + {q4_col}')

# Load Netflix actual published data
print()
print('[3] Loading Netflix published actuals...')

as_published = ORIGINALS / 'As Published'

# Map Netflix files to time periods
netflix_periods = {
    'What_We_Watched_A_Netflix_Engagement_Report_2024Jan-Jun.xlsx': ('h1', '2024'),
    'What_We_Watched_A_Netflix_Engagement_Report_2024Jul-Dec.xlsx': ('h2', '2024'),
    'What_We_Watched_A_Netflix_Engagement_Report_2025Jan-Jun.xlsx': ('h1', '2025'),
    'What_We_Watched_A_Netflix_Engagement_Report_2025Jul-Dec__6_.xlsx': ('h2', '2025'),
}

all_comparisons = []

for filename, (period, year) in netflix_periods.items():
    filepath = as_published / filename
    if not filepath.exists():
        print(f'    [SKIP] {filename} not found')
        continue

    print(f'    Loading: {filename} -> {period.upper()} {year}')

    nf = pd.read_excel(filepath, skiprows=5)

    # Find title and views columns
    title_col = 'Title' if 'Title' in nf.columns else None
    views_col = 'Views' if 'Views' in nf.columns else None

    if not title_col or not views_col:
        print(f'      [ERROR] Missing columns: {list(nf.columns)}')
        continue

    nf_clean = nf[[title_col, views_col]].dropna()
    nf_clean.columns = ['netflix_title', 'netflix_views']
    nf_clean['netflix_views'] = pd.to_numeric(nf_clean['netflix_views'], errors='coerce')
    nf_clean = nf_clean[nf_clean['netflix_views'] > 0]

    print(f'      Titles: {len(nf_clean):,}')

    # Clean Netflix titles for matching
    def clean_title(t):
        t = str(t).lower().strip()
        # Remove season/series suffixes for matching
        t = re.sub(r':\s*(season\s*\d+|limited\s*series|series\s*\d+).*$', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s+', ' ', t)
        return t.strip()

    nf_clean['title_clean'] = nf_clean['netflix_title'].apply(clean_title)

    # Get the corresponding FlixPatrol column
    fp_col = f'views_{period}_{year}_total'

    if fp_col not in db.columns:
        print(f'      [ERROR] Column {fp_col} not in database')
        continue

    # Clean database titles
    db['title_clean'] = db['title'].apply(clean_title)

    # Merge on cleaned title
    merged = nf_clean.merge(
        db[['title_clean', fp_col, 'title', 'fc_uid']],
        on='title_clean',
        how='inner'
    )

    merged = merged[merged[fp_col] > 0]

    print(f'      Matched: {len(merged):,}')

    if len(merged) > 0:
        # Calculate APE
        merged['ape'] = np.abs(merged[fp_col] - merged['netflix_views']) / merged['netflix_views']
        merged['period'] = f'{period.upper()} {year}'
        merged['fp_views'] = merged[fp_col]
        all_comparisons.append(merged[['netflix_title', 'title', 'fc_uid', 'netflix_views', 'fp_views', 'ape', 'period']])

# Combine all comparisons
if all_comparisons:
    all_data = pd.concat(all_comparisons, ignore_index=True)

    # Filter valid APE (< 1000%)
    valid = all_data[all_data['ape'] < 10]

    print()
    print('='*80)
    print('MAPE RESULTS')
    print('='*80)
    print(f'  Total Comparisons:  {len(all_data):,}')
    print(f'  Valid Comparisons:  {len(valid):,}')

    mape = valid['ape'].mean() * 100
    median_ape = valid['ape'].median() * 100

    print()
    print(f'  MAPE:               {mape:.2f}%')
    print(f'  Median APE:         {median_ape:.2f}%')

    # Correlation
    corr = valid['fp_views'].corr(valid['netflix_views'])
    print(f'  Correlation (r):    {corr:.4f}')
    print(f'  R-squared:          {corr**2:.4f}')

    # By period
    print()
    print('  MAPE by Period:')
    for period in valid['period'].unique():
        period_data = valid[valid['period'] == period]
        period_mape = period_data['ape'].mean() * 100
        print(f'    {period}: {period_mape:.2f}% ({len(period_data):,} titles)')

    # Error distribution
    print()
    print('  Error Distribution:')
    ranges = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 500), (500, 1000)]
    for low, high in ranges:
        count = ((valid['ape']*100 >= low) & (valid['ape']*100 < high)).sum()
        pct = count / len(valid) * 100
        print(f'    {low:>4}% - {high:<4}%: {count:>6,} titles ({pct:>5.1f}%)')

    # Best matches
    print()
    print('  Top 10 Best Matches (Lowest APE):')
    top = valid.nsmallest(10, 'ape')
    for _, row in top.iterrows():
        print(f'    {row["netflix_title"][:35]:<35} NF:{row["netflix_views"]:>12,.0f} FP:{row["fp_views"]:>12,.0f} APE:{row["ape"]*100:>6.1f}%')

    # Anti-cheat
    print()
    print('='*80)
    print('ANTI-CHEAT VALIDATION')
    print('='*80)
    print(f'  MAPE Valid Range:   5% - 40%')
    print(f'  Actual MAPE:        {mape:.2f}%')
    status = 'PASS' if 5.0 <= mape <= 40.0 else 'REVIEW'
    print(f'  Status:             {status}')
    print('='*80)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_comparisons': len(all_data),
        'valid_comparisons': len(valid),
        'mape_percent': round(mape, 2),
        'median_ape_percent': round(median_ape, 2),
        'correlation': round(corr, 4),
        'r_squared': round(corr**2, 4),
        'by_period': {p: {'mape': round(valid[valid['period']==p]['ape'].mean()*100, 2),
                         'count': len(valid[valid['period']==p])}
                     for p in valid['period'].unique()},
        'anti_cheat_check': {
            'mape_in_valid_range': bool(5.0 <= mape <= 40.0),
            'valid_range': '5% - 40%'
        }
    }

    output_file = BASE / 'MAPIE Engine' / f'MAPE_SCORES_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved: {output_file}')
else:
    print('\nERROR: No comparisons generated')
