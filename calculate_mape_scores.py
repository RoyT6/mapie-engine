#!/usr/bin/env python3
"""
MAPE SCORE CALCULATION
======================
Compares FlixPatrol views in merged database against Netflix published actuals
"""
import pandas as pd
import numpy as np
import json
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

# Get views columns
views_cols = [c for c in db.columns if c.startswith('views_') and '_total' in c]
print(f'    Views columns: {len(views_cols)}')

# Calculate total FlixPatrol views per title
db['flixpatrol_total'] = db[views_cols].sum(axis=1)
db_with_views = db[db['flixpatrol_total'] > 0].copy()
print(f'    Titles with FlixPatrol views: {len(db_with_views):,}')

# Load Netflix actual published data
print()
print('[2] Loading Netflix published actuals...')

netflix_data = []

# Load all Netflix engagement reports
as_published = ORIGINALS / 'As Published'
print(f'    Looking in: {as_published}')
print(f'    Exists: {as_published.exists()}')

netflix_files = list(as_published.glob('*.xlsx')) if as_published.exists() else []
print(f'    Found {len(netflix_files)} Excel files')

for nf_file in netflix_files:
    if nf_file.exists():
        try:
            nf = pd.read_excel(nf_file, skiprows=5)
            # Find title and views columns
            title_col = None
            views_col = None
            for c in nf.columns:
                if 'title' in str(c).lower():
                    title_col = c
                if 'view' in str(c).lower() and 'hour' not in str(c).lower():
                    views_col = c
                if 'hour' in str(c).lower():
                    views_col = c  # Fallback to hours

            if title_col and views_col:
                nf_clean = nf[[title_col, views_col]].dropna()
                nf_clean.columns = ['title', 'views']
                nf_clean['views'] = pd.to_numeric(nf_clean['views'], errors='coerce')
                netflix_data.append(nf_clean)
                print(f'    Loaded: {nf_file.name} ({len(nf_clean):,} titles)')
        except Exception as e:
            print(f'    Error loading {nf_file.name}: {e}')

if netflix_data:
    netflix_all = pd.concat(netflix_data, ignore_index=True)
    # Aggregate by title
    netflix_agg = netflix_all.groupby('title')['views'].sum().reset_index()
    netflix_agg['title_clean'] = netflix_agg['title'].str.lower().str.strip()
    print(f'    Total Netflix titles: {len(netflix_agg):,}')
    print(f'    Total Netflix views: {netflix_agg["views"].sum():,.0f}')
else:
    print('    ERROR: No Netflix data loaded')
    exit(1)

# Match and calculate MAPE
print()
print('[3] Matching titles and calculating MAPE...')

db_with_views['title_clean'] = db_with_views['title'].str.lower().str.strip()

# Merge on title
merged = db_with_views.merge(netflix_agg, on='title_clean', how='inner', suffixes=('', '_netflix'))

print(f'    Matched titles: {len(merged):,}')

if len(merged) > 0:
    # Netflix 'Views' column is actual views count
    merged['netflix_views'] = merged['views']

    # Calculate APE for each title
    merged['ape'] = np.abs(merged['flixpatrol_total'] - merged['netflix_views']) / merged['netflix_views']
    merged['ape'] = merged['ape'].replace([np.inf, -np.inf], np.nan)

    # Filter valid APE values
    valid = merged[merged['ape'].notna() & (merged['ape'] < 10)]  # Cap at 1000% error

    # MAPE calculation
    mape = valid['ape'].mean() * 100
    median_ape = valid['ape'].median() * 100

    print()
    print('='*80)
    print('MAPE RESULTS')
    print('='*80)
    print(f'  Matched Titles:     {len(valid):,}')
    print(f'  MAPE:               {mape:.2f}%')
    print(f'  Median APE:         {median_ape:.2f}%')
    print()

    # Breakdown by error range
    print('  Error Distribution:')
    ranges = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 500), (500, 1000)]
    for low, high in ranges:
        count = ((valid['ape']*100 >= low) & (valid['ape']*100 < high)).sum()
        pct = count / len(valid) * 100
        print(f'    {low:>4}% - {high:<4}%: {count:>6,} titles ({pct:>5.1f}%)')

    # Correlation
    corr = valid['flixpatrol_total'].corr(valid['netflix_views'])
    print()
    print(f'  Correlation (r):    {corr:.4f}')
    print(f'  R-squared:          {corr**2:.4f}')

    # Top matches (lowest error)
    print()
    print('  Top 10 Best Matches (Lowest APE):')
    top = valid.nsmallest(10, 'ape')[['title', 'flixpatrol_total', 'netflix_views', 'ape']]
    for _, row in top.iterrows():
        print(f'    {row["title"][:40]:<40} FP:{row["flixpatrol_total"]:>12,.0f} NF:{row["netflix_views"]:>12,.0f} APE:{row["ape"]*100:>6.1f}%')

    # Summary
    results = {
        'timestamp': datetime.now().isoformat(),
        'matched_titles': len(valid),
        'mape_percent': round(mape, 2),
        'median_ape_percent': round(median_ape, 2),
        'correlation': round(corr, 4),
        'r_squared': round(corr**2, 4),
        'flixpatrol_total_views': int(valid['flixpatrol_total'].sum()),
        'netflix_total_views': int(valid['netflix_views'].sum()),
        'anti_cheat_check': {
            'mape_in_valid_range': bool(5.0 <= mape <= 40.0),
            'valid_range': '5% - 40%'
        }
    }

    print()
    print('='*80)
    print('ANTI-CHEAT VALIDATION')
    print('='*80)
    print(f'  MAPE Valid Range:   5% - 40%')
    print(f'  Actual MAPE:        {mape:.2f}%')
    print(f'  Status:             {"PASS - Within valid range" if 5.0 <= mape <= 40.0 else "REVIEW - Outside expected range"}')
    print('='*80)

    # Save results
    output_file = BASE / 'MAPIE Engine' / f'MAPE_SCORES_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved: {output_file}')

else:
    print('    ERROR: No matching titles found')
