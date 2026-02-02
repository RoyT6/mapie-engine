#!/usr/bin/env python3
"""
MAPE SCORE CALCULATION - FIXED FOR KILLER ISSUE (SEASONS)
==========================================================
Properly matches Netflix title+season to database fc_uid
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
print('MAPE SCORE CALCULATION - Season-Aware Matching')
print('='*80)
print(f'Timestamp: {datetime.now().isoformat()}')
print()

BASE = Path('C:/Users/RoyT6/Downloads')
ORIGINALS = Path('C:/Users/RoyT6/Downloads/Training Data/Originals')

# Load merged database
print('[1] Loading merged database...')
db = pd.read_parquet(BASE / 'BFD-Views-2026-Feb-2.00.parquet')
print(f'    Rows: {len(db):,} | Columns: {len(db.columns)}')

# Parse season from fc_uid for matching
def extract_season_from_fcuid(fc_uid):
    """Extract season number from fc_uid like tt5180504_s03 -> 3"""
    if pd.isna(fc_uid):
        return None
    match = re.search(r'_s(\d+)$', str(fc_uid))
    if match:
        return int(match.group(1))
    return None  # Films have no season

db['db_season'] = db['fc_uid'].apply(extract_season_from_fcuid)
db['title_clean'] = db['title'].str.lower().str.strip()

print(f'    TV shows (with season): {db["db_season"].notna().sum():,}')
print(f'    Films (no season): {db["db_season"].isna().sum():,}')

# Load Netflix data and parse seasons
print()
print('[2] Loading Netflix published actuals with season parsing...')

as_published = ORIGINALS / 'As Published'

netflix_periods = {
    'What_We_Watched_A_Netflix_Engagement_Report_2024Jan-Jun.xlsx': ('h1', '2024'),
    'What_We_Watched_A_Netflix_Engagement_Report_2024Jul-Dec.xlsx': ('h2', '2024'),
    'What_We_Watched_A_Netflix_Engagement_Report_2025Jan-Jun.xlsx': ('h1', '2025'),
    'What_We_Watched_A_Netflix_Engagement_Report_2025Jul-Dec__6_.xlsx': ('h2', '2025'),
}

def parse_netflix_title(title):
    """
    Parse Netflix title to extract clean title and season number.
    Examples:
        'The Witcher: Season 3' -> ('the witcher', 3)
        'Fool Me Once: Limited Series' -> ('fool me once', 1)
        'Happy Gilmore 2' -> ('happy gilmore 2', None)  # Film
    """
    title_str = str(title).strip()

    # Extract season number patterns
    season_patterns = [
        r':\s*Season\s*(\d+)',           # : Season 3
        r':\s*Series\s*(\d+)',            # : Series 2
        r':\s*Part\s*(\d+)',              # : Part 2
        r':\s*Volume\s*(\d+)',            # : Volume 3
        r':\s*Chapter\s*(\d+)',           # : Chapter 4
    ]

    season_num = None
    clean_title = title_str

    for pattern in season_patterns:
        match = re.search(pattern, title_str, re.IGNORECASE)
        if match:
            season_num = int(match.group(1))
            clean_title = re.sub(pattern, '', title_str, flags=re.IGNORECASE)
            break

    # Check for Limited Series (treat as season 1)
    if re.search(r':\s*Limited\s*Series', title_str, re.IGNORECASE):
        season_num = 1
        clean_title = re.sub(r':\s*Limited\s*Series.*$', '', title_str, flags=re.IGNORECASE)

    # Clean up the title
    clean_title = clean_title.strip().rstrip(':').strip().lower()

    return clean_title, season_num

all_comparisons = []

for filename, (period, year) in netflix_periods.items():
    filepath = as_published / filename
    if not filepath.exists():
        continue

    print(f'    Loading: {filename} -> {period.upper()} {year}')

    nf = pd.read_excel(filepath, skiprows=5)

    if 'Title' not in nf.columns or 'Views' not in nf.columns:
        print(f'      [ERROR] Missing columns')
        continue

    nf_clean = nf[['Title', 'Views']].dropna()
    nf_clean.columns = ['netflix_title', 'netflix_views']
    nf_clean['netflix_views'] = pd.to_numeric(nf_clean['netflix_views'], errors='coerce')
    nf_clean = nf_clean[nf_clean['netflix_views'] > 0]

    # Parse title and season
    parsed = nf_clean['netflix_title'].apply(parse_netflix_title)
    nf_clean['title_clean'] = parsed.apply(lambda x: x[0])
    nf_clean['nf_season'] = parsed.apply(lambda x: x[1])

    tv_count = nf_clean['nf_season'].notna().sum()
    film_count = nf_clean['nf_season'].isna().sum()
    print(f'      Titles: {len(nf_clean):,} (TV seasons: {tv_count:,}, Films: {film_count:,})')

    # Get corresponding views column
    fp_col = f'views_{period}_{year}_total'
    if fp_col not in db.columns:
        # Try constructing from quarters
        if period == 'h1':
            q1 = f'views_q1_{year}_total'
            q2 = f'views_q2_{year}_total'
            if q1 in db.columns and q2 in db.columns:
                db[fp_col] = db[q1].fillna(0) + db[q2].fillna(0)
        else:
            q3 = f'views_q3_{year}_total'
            q4 = f'views_q4_{year}_total'
            if q3 in db.columns and q4 in db.columns:
                db[fp_col] = db[q3].fillna(0) + db[q4].fillna(0)

    if fp_col not in db.columns:
        print(f'      [ERROR] Column {fp_col} not available')
        continue

    # Match: title + season for TV, title only for films
    # TV shows: match on title_clean + season
    tv_netflix = nf_clean[nf_clean['nf_season'].notna()].copy()
    tv_db = db[db['db_season'].notna()].copy()

    tv_merged = tv_netflix.merge(
        tv_db[['title_clean', 'db_season', fp_col, 'fc_uid', 'title']],
        left_on=['title_clean', 'nf_season'],
        right_on=['title_clean', 'db_season'],
        how='inner'
    )
    tv_merged = tv_merged[tv_merged[fp_col] > 0]

    # Films: match on title_clean only (no season)
    film_netflix = nf_clean[nf_clean['nf_season'].isna()].copy()
    film_db = db[db['db_season'].isna()].copy()

    film_merged = film_netflix.merge(
        film_db[['title_clean', fp_col, 'fc_uid', 'title']],
        on='title_clean',
        how='inner'
    )
    film_merged = film_merged[film_merged[fp_col] > 0]

    print(f'      Matched: TV={len(tv_merged):,}, Films={len(film_merged):,}')

    # Combine and calculate APE
    for merged_df, content_type in [(tv_merged, 'TV'), (film_merged, 'Film')]:
        if len(merged_df) > 0:
            merged_df = merged_df.copy()
            merged_df['ape'] = np.abs(merged_df[fp_col] - merged_df['netflix_views']) / merged_df['netflix_views']
            merged_df['period'] = f'{period.upper()} {year}'
            merged_df['content_type'] = content_type
            merged_df['fp_views'] = merged_df[fp_col]
            all_comparisons.append(merged_df[['netflix_title', 'title', 'fc_uid', 'netflix_views', 'fp_views', 'ape', 'period', 'content_type']])

# Combine all comparisons
if all_comparisons:
    all_data = pd.concat(all_comparisons, ignore_index=True)

    # Filter valid APE (< 1000%)
    valid = all_data[all_data['ape'] < 10]

    print()
    print('='*80)
    print('MAPE RESULTS - SEASON-AWARE MATCHING')
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

    # By content type
    print()
    print('  MAPE by Content Type:')
    for ct in valid['content_type'].unique():
        ct_data = valid[valid['content_type'] == ct]
        ct_mape = ct_data['ape'].mean() * 100
        print(f'    {ct}: {ct_mape:.2f}% ({len(ct_data):,} titles)')

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
        print(f'    {row["netflix_title"][:40]:<40} NF:{row["netflix_views"]:>12,.0f} DB:{row["fp_views"]:>12,.0f} APE:{row["ape"]*100:>6.2f}%')

    # Anti-cheat validation
    print()
    print('='*80)
    print('ANTI-CHEAT VALIDATION')
    print('='*80)
    print(f'  MAPE Valid Range:   5% - 40%')
    print(f'  Actual MAPE:        {mape:.2f}%')
    status = 'PASS' if 5.0 <= mape <= 40.0 else ('REVIEW' if mape < 5.0 else 'HIGH')
    print(f'  Status:             {status}')
    print('='*80)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'methodology': 'Season-aware matching (title + season_number)',
        'total_comparisons': len(all_data),
        'valid_comparisons': len(valid),
        'mape_percent': round(mape, 2),
        'median_ape_percent': round(median_ape, 2),
        'correlation': round(corr, 4),
        'r_squared': round(corr**2, 4),
        'by_content_type': {ct: {'mape': round(valid[valid['content_type']==ct]['ape'].mean()*100, 2),
                                 'count': len(valid[valid['content_type']==ct])}
                           for ct in valid['content_type'].unique()},
        'by_period': {p: {'mape': round(valid[valid['period']==p]['ape'].mean()*100, 2),
                         'count': len(valid[valid['period']==p])}
                     for p in valid['period'].unique()},
        'anti_cheat_check': {
            'mape_in_valid_range': bool(5.0 <= mape <= 40.0),
            'valid_range': '5% - 40%',
            'status': status
        }
    }

    output_file = BASE / 'MAPIE Engine' / f'MAPE_SCORES_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved: {output_file}')
else:
    print('\nERROR: No comparisons generated')
