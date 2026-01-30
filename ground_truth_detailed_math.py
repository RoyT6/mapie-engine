#!/usr/bin/env python3
"""
GROUND TRUTH DETAILED MATH - NO EXCUSES
Shows exact computed vs actual values for every matched title
"""
import os
os.environ['CUDF_SPILL'] = 'on'

import cudf
import pandas as pd
import numpy as np
from datetime import datetime

print('='*100)
print('GROUND TRUTH DETAILED MATH - FULL TRANSPARENCY')
print('='*100)
print(f'Timestamp: {datetime.now().isoformat()}')
print()

BASE = '/mnt/c/Users/RoyT6/Downloads'
VIEWS_TRAINING = f'{BASE}/Views TRaining Data'

# Load BFD
print('[1] Loading computed views from BFD...')
bfd = cudf.read_parquet(f'{BASE}/Cranberry_BFD_MAPIE_RUN_20260116_V19.87.parquet')
bfd_pd = bfd[['imdb_id', 'title', 'views_computed']].to_pandas()
del bfd
print(f'    Loaded: {len(bfd_pd):,} titles')
print(f'    views_computed stats:')
print(f'      Min: {bfd_pd["views_computed"].min():,}')
print(f'      Max: {bfd_pd["views_computed"].max():,}')
print(f'      Mean: {bfd_pd["views_computed"].mean():,.0f}')
print(f'      Median: {bfd_pd["views_computed"].median():,.0f}')

# Load AGGREGATED_VIEWS_BY_IMDB
print()
print('[2] Loading actual views from AGGREGATED_VIEWS_BY_IMDB.csv...')
agg = pd.read_csv(f'{VIEWS_TRAINING}/AGGREGATED_VIEWS_BY_IMDB.csv')
print(f'    Columns: {list(agg.columns)}')
print(f'    Rows: {len(agg):,}')
print()
print('    First 5 rows:')
print(agg.head().to_string())

# Find the views column
views_col = None
for col in agg.columns:
    if 'view' in col.lower():
        views_col = col
        print(f'\n    Using views column: "{views_col}"')
        break

if views_col is None:
    print('    ERROR: No views column found!')
    print(f'    Available columns: {list(agg.columns)}')
    exit(1)

print(f'\n    Actual views stats ({views_col}):')
print(f'      Min: {agg[views_col].min():,}')
print(f'      Max: {agg[views_col].max():,}')
print(f'      Mean: {agg[views_col].mean():,.0f}')
print(f'      Median: {agg[views_col].median():,.0f}')

# Find IMDB ID column
id_col = None
for col in agg.columns:
    if 'imdb' in col.lower():
        id_col = col
        print(f'    Using ID column: "{id_col}"')
        break

# Merge
print()
print('[3] Merging computed vs actual...')
bfd_pd['imdb_id'] = bfd_pd['imdb_id'].astype(str).str.strip()
agg[id_col] = agg[id_col].astype(str).str.strip()

merged = bfd_pd.merge(
    agg[[id_col, views_col]].rename(columns={id_col: 'imdb_id', views_col: 'actual_views'}),
    on='imdb_id',
    how='inner'
)
print(f'    Matched: {len(merged):,} titles')

# Filter to valid actual views
merged = merged[merged['actual_views'] > 0].copy()
print(f'    With actual_views > 0: {len(merged):,} titles')

# Calculate individual errors
print()
print('[4] CALCULATING ERRORS FOR EVERY TITLE')
print('='*100)

merged['error'] = merged['views_computed'] - merged['actual_views']
merged['abs_error'] = merged['error'].abs()
merged['pct_error'] = (merged['error'] / merged['actual_views']) * 100
merged['abs_pct_error'] = merged['pct_error'].abs()

# Sort by actual views descending to show most important titles first
merged = merged.sort_values('actual_views', ascending=False)

print()
print('TOP 50 TITLES BY ACTUAL VIEWS (Computed vs Actual):')
print('-'*100)
print(f'{"Title":<40} {"Computed":>15} {"Actual":>15} {"Error":>15} {"% Error":>12}')
print('-'*100)

for i, row in merged.head(50).iterrows():
    title = str(row['title'])[:38]
    computed = row['views_computed']
    actual = row['actual_views']
    error = row['error']
    pct_err = row['pct_error']
    print(f'{title:<40} {computed:>15,.0f} {actual:>15,.0f} {error:>+15,.0f} {pct_err:>+11.1f}%')

print('-'*100)

print()
print('='*100)
print('[5] FULL ERROR DISTRIBUTION')
print('='*100)

print(f'\nTotal matched titles: {len(merged):,}')
print()
print('ERROR STATISTICS:')
print(f'  Mean Error:        {merged["error"].mean():>+20,.0f}')
print(f'  Median Error:      {merged["error"].median():>+20,.0f}')
print(f'  Std Dev:           {merged["error"].std():>20,.0f}')
print()
print('ABSOLUTE PERCENT ERROR STATISTICS:')
print(f'  Mean APE (MAPE):   {merged["abs_pct_error"].mean():>19.2f}%')
print(f'  Median APE:        {merged["abs_pct_error"].median():>19.2f}%')
print(f'  Std Dev:           {merged["abs_pct_error"].std():>19.2f}%')
print(f'  Min APE:           {merged["abs_pct_error"].min():>19.2f}%')
print(f'  Max APE:           {merged["abs_pct_error"].max():>19.2f}%')

print()
print('MAPE CALCULATION (Manual Verification):')
print('-'*60)
print('  MAPE = (1/n) * Σ |computed - actual| / actual * 100')
print()
print(f'  n = {len(merged):,}')
print(f'  Σ |computed - actual| / actual = {merged["abs_pct_error"].sum()/100:.4f}')
print(f'  MAPE = {merged["abs_pct_error"].sum()/100 / len(merged) * 100:.2f}%')

print()
print('ERROR DISTRIBUTION BY PERCENTILE:')
print('-'*60)
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(merged['abs_pct_error'], p)
    print(f'  {p:>2}th percentile: {val:>10.2f}% APE')

print()
print('ERROR DIRECTION:')
print('-'*60)
over = (merged['error'] > 0).sum()
under = (merged['error'] < 0).sum()
exact = (merged['error'] == 0).sum()
print(f'  Over-estimated:   {over:>6,} ({over/len(merged)*100:.1f}%)')
print(f'  Under-estimated:  {under:>6,} ({under/len(merged)*100:.1f}%)')
print(f'  Exact:            {exact:>6,} ({exact/len(merged)*100:.1f}%)')

print()
print('SCALE ANALYSIS:')
print('-'*60)
print(f'  Computed views range: {merged["views_computed"].min():>15,.0f} - {merged["views_computed"].max():>15,.0f}')
print(f'  Actual views range:   {merged["actual_views"].min():>15,.0f} - {merged["actual_views"].max():>15,.0f}')
print(f'  Ratio (computed/actual) mean:   {(merged["views_computed"]/merged["actual_views"]).mean():.4f}')
print(f'  Ratio (computed/actual) median: {(merged["views_computed"]/merged["actual_views"]).median():.4f}')

print()
print('CORRELATION ANALYSIS:')
print('-'*60)
corr = merged['views_computed'].corr(merged['actual_views'])
print(f'  Pearson correlation: {corr:.4f}')
print(f'  R² (coefficient of determination): {corr**2:.4f}')

# Spearman (rank correlation)
spearman = merged['views_computed'].rank().corr(merged['actual_views'].rank())
print(f'  Spearman rank correlation: {spearman:.4f}')

print()
print('='*100)
print('[6] WORST 20 ERRORS (Highest APE)')
print('='*100)
worst = merged.nlargest(20, 'abs_pct_error')
print(f'{"Title":<40} {"Computed":>15} {"Actual":>15} {"APE":>12}')
print('-'*85)
for i, row in worst.iterrows():
    title = str(row['title'])[:38]
    print(f'{title:<40} {row["views_computed"]:>15,.0f} {row["actual_views"]:>15,.0f} {row["abs_pct_error"]:>11.1f}%')

print()
print('='*100)
print('[7] BEST 20 ERRORS (Lowest APE)')
print('='*100)
best = merged.nsmallest(20, 'abs_pct_error')
print(f'{"Title":<40} {"Computed":>15} {"Actual":>15} {"APE":>12}')
print('-'*85)
for i, row in best.iterrows():
    title = str(row['title'])[:38]
    print(f'{title:<40} {row["views_computed"]:>15,.0f} {row["actual_views"]:>15,.0f} {row["abs_pct_error"]:>11.1f}%')

print()
print('='*100)
print('[8] VERDICT')
print('='*100)
mape = merged['abs_pct_error'].mean()
print()
print(f'  MAPE = {mape:.2f}%')
print()
if mape > 50:
    print('  STATUS: FAIL - MAPE > 50% indicates computed views do not match actual views')
    print('  THE MODEL IS NOT PREDICTING ACTUAL VIEWERSHIP ACCURATELY')
elif mape > 25:
    print('  STATUS: POOR - MAPE 25-50% indicates weak predictive accuracy')
else:
    print('  STATUS: ACCEPTABLE - MAPE < 25%')

print()
print('  TRUTH: An 82% MAPE means on average, our computed views are')
print('         82% different from the actual published views.')
print('         This is NOT good. No excuses.')
print()
print('='*100)

# Save detailed results
merged.to_csv(f'{BASE}/MAPIE/ground_truth_comparison_detailed.csv', index=False)
print(f'\nFull comparison saved to: MAPIE/ground_truth_comparison_detailed.csv')
