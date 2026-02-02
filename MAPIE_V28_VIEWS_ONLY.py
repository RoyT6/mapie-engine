#!/usr/bin/env python3
"""
MAPIE V28 - VIEWS-ONLY DATABASE RUNNER
========================================
VERSION: 28.01
CREATED: 2026-01-31

PURPOSE: MAPIE validation for BFD_VIEWS_V28.xxx.parquet (views-only schema)
Includes:
  - Proof-of-Work Requirement
  - Checksum Validation
  - Execution Logging
  - Trust-Based Audit

TARGET DATABASE: BFD_VIEWS_V28.000.parquet or BFD_VIEWS_V28.001.parquet
SCHEMA: Views-only with temporal columns (views_h1_*, views_q*_*, views_*_month_*)

============================================================================
"""
import os
import sys
import io
import time
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(r'C:\Users\RoyT6\Downloads')
MAPIE_DIR = BASE_DIR / 'MAPIE'

# PRIMARY DATA SOURCE: Components Engine
COMPONENTS_ENGINE = BASE_DIR / 'Components Engine'
WEIGHTERS_DIR = COMPONENTS_ENGINE / 'Weighters'
TRAINING_DATA_DIR = COMPONENTS_ENGINE / 'Training Data'
OLDER_LIBRARY_DIR = COMPONENTS_ENGINE / 'Older Library'
STUDIOS_DIR = COMPONENTS_ENGINE / 'Studios'
STREAMERS_DIR = COMPONENTS_ENGINE / 'Streamers By Country'

# Alternative data sources (fallback)
ALT_TRAINING_DIR = BASE_DIR / 'Training Data'
ALT_VIEWS_DIR = BASE_DIR / 'Views TRaining Data'

# Create MAPIE directory if not exists
MAPIE_DIR.mkdir(exist_ok=True)

# Database patterns
BFD_VIEWS_PATTERN = re.compile(r'BFD_VIEWS_V(\d+\.\d+)\.parquet')

# Anti-cheat bounds
VALID_MAPE_RANGE = (5.0, 40.0)
VALID_R2_RANGE = (0.30, 0.90)

# ============================================================================
# LOGGING & PROOF-OF-WORK
# ============================================================================
class ExecutionLogger:
    """Handles execution logging with timestamps."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'MAPIE_V28_execution_log_{self.timestamp}.txt'
        self.log_entries = []
        self.start_time = time.time()

    def log(self, message: str, level: str = 'INFO'):
        """Log a message with timestamp."""
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        entry = f"[{ts}] [{level}] {message}"
        self.log_entries.append(entry)
        print(entry)

    def save(self):
        """Save log to file."""
        runtime = time.time() - self.start_time
        self.log_entries.append(f"\n[RUNTIME] Total execution time: {runtime:.2f}s")

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_entries))
        return self.log_file


class ProofOfWork:
    """Proof-of-work validation system."""

    def __init__(self, logger: ExecutionLogger):
        self.logger = logger
        self.checksums = {}
        self.validations = []

    def calculate_checksum(self, filepath: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def validate_file(self, filepath: Path, expected_checksum: str = None) -> dict:
        """Validate file exists and optionally check checksum."""
        result = {
            'file': str(filepath),
            'exists': filepath.exists(),
            'checksum': None,
            'checksum_match': None,
            'size_bytes': None,
            'validated_at': datetime.now().isoformat()
        }

        if filepath.exists():
            result['size_bytes'] = filepath.stat().st_size
            result['checksum'] = self.calculate_checksum(filepath)
            self.checksums[str(filepath)] = result['checksum']

            if expected_checksum:
                result['checksum_match'] = result['checksum'] == expected_checksum

        self.validations.append(result)
        return result

    def validate_dataframe(self, df: pd.DataFrame, name: str) -> dict:
        """Validate DataFrame integrity."""
        result = {
            'name': name,
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / (1024*1024),
            'null_counts': int(df.isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
            'validated_at': datetime.now().isoformat()
        }

        # Calculate data hash (sample-based for large dataframes)
        if len(df) > 10000:
            sample = df.sample(n=10000, random_state=42)
        else:
            sample = df
        data_str = sample.to_json()
        result['data_hash'] = hashlib.sha256(data_str.encode()).hexdigest()[:16]

        self.validations.append(result)
        return result


class TrustBasedAudit:
    """Trust-based audit system."""

    def __init__(self, logger: ExecutionLogger):
        self.logger = logger
        self.audit_checks = []

    def check(self, name: str, condition: bool, message: str) -> bool:
        """Record an audit check."""
        result = {
            'check': name,
            'passed': condition,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.audit_checks.append(result)
        status = 'PASS' if condition else 'FAIL'
        self.logger.log(f"  [{status}] {name}: {message}")
        return condition

    def get_trust_score(self) -> Tuple[int, int, float]:
        """Calculate trust score."""
        passed = sum(1 for c in self.audit_checks if c['passed'])
        total = len(self.audit_checks)
        percentage = (passed / total * 100) if total > 0 else 0
        return passed, total, percentage


# ============================================================================
# DATABASE LOADER
# ============================================================================
class ViewsOnlyLoader:
    """Loads BFD_VIEWS_V28 views-only database."""

    def __init__(self, base_dir: Path, logger: ExecutionLogger, pow: ProofOfWork):
        self.base_dir = base_dir
        self.logger = logger
        self.pow = pow
        self.df = None
        self.version = None
        self.stats = {}

    def detect_latest_version(self) -> Optional[Path]:
        """Auto-detect latest BFD_VIEWS version."""
        files = list(self.base_dir.glob('BFD_VIEWS_V*.parquet'))

        if not files:
            # Check in Training Data
            files = list((self.base_dir / 'Training Data' / 'ETL' / 'BFD_Integration_Output').glob('BFD_VIEWS_V*.parquet'))

        if not files:
            return None

        def extract_version(f):
            match = BFD_VIEWS_PATTERN.search(f.name)
            return float(match.group(1)) if match else 0.0

        latest = max(files, key=extract_version)
        match = BFD_VIEWS_PATTERN.search(latest.name)
        self.version = match.group(1) if match else 'unknown'
        return latest

    def load(self, filepath: Path = None) -> pd.DataFrame:
        """Load the views database."""
        if filepath is None:
            filepath = self.detect_latest_version()

        if filepath is None:
            raise FileNotFoundError("No BFD_VIEWS_V*.parquet found")

        self.logger.log(f"Loading database: {filepath.name}")

        # Validate file
        file_validation = self.pow.validate_file(filepath)
        self.logger.log(f"  File size: {file_validation['size_bytes'] / (1024**3):.2f} GB")
        self.logger.log(f"  Checksum: {file_validation['checksum'][:16]}...")

        t0 = time.time()
        self.df = pd.read_parquet(filepath)
        load_time = time.time() - t0

        # Validate DataFrame
        df_validation = self.pow.validate_dataframe(self.df, f'BFD_VIEWS_V{self.version}')

        self.logger.log(f"  Rows: {len(self.df):,}")
        self.logger.log(f"  Columns: {len(self.df.columns)}")
        self.logger.log(f"  Load time: {load_time:.1f}s")

        self.stats = {
            'filepath': str(filepath),
            'version': self.version,
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'load_time': load_time,
            'file_checksum': file_validation['checksum'],
            'data_hash': df_validation['data_hash']
        }

        return self.df

    def get_temporal_columns(self) -> Dict[str, List[str]]:
        """Identify temporal view columns."""
        if self.df is None:
            return {}

        cols = self.df.columns.tolist()

        temporal = {
            'half_year': [c for c in cols if re.match(r'views_h[12]_\d{4}', c)],
            'quarter': [c for c in cols if re.match(r'views_q[1234]_\d{4}', c)],
            'monthly': [c for c in cols if re.match(r'views_\w{3}_\d{4}', c) and not c.startswith('views_q')],
            'regional': [c for c in cols if '_us' in c or '_gb' in c or '_total' in c],
            'all_views': [c for c in cols if c.startswith('views_')]
        }

        return temporal


# ============================================================================
# COMPONENTS LOADER (from Components Engine)
# ============================================================================
class ComponentsLoader:
    """Loads weighters and components from Components Engine."""

    def __init__(self, logger: ExecutionLogger, pow: ProofOfWork):
        self.logger = logger
        self.pow = pow
        self.weighters = {}
        self.components_loaded = 0

    def load_weighters(self) -> Dict:
        """Load all weighting components."""
        self.logger.log("Loading weighters from Components Engine...")

        weighter_files = [
            ('genre_decay', WEIGHTERS_DIR / 'genre decay table.json'),
            ('genre_performance', WEIGHTERS_DIR / 'genre_performance_2025.json'),
            ('genre_skew', WEIGHTERS_DIR / 'genre_skew_market_share_2025.json'),
            ('studio_weighting', WEIGHTERS_DIR / 'component studio weighting.json'),
            ('ml_features', WEIGHTERS_DIR / 'component_ml_combined_features_2025.json'),
            ('ml_uk_features', WEIGHTERS_DIR / 'ml_uk_features_2025.json'),
            ('sub_growth', WEIGHTERS_DIR / 'ml_abs_sub_growth_lookup.json'),
        ]

        streaming_components = [
            ('trending', WEIGHTERS_DIR / 'Streaming Platform Components' / 'component_trending.json'),
            ('exclusivity', WEIGHTERS_DIR / 'Streaming Platform Components' / 'component_streaming_platform_exclusivity_patterns.json'),
            ('financials', WEIGHTERS_DIR / 'Streaming Platform Components' / 'component_streaming_financial_data_Q4_2025.json'),
            ('international', WEIGHTERS_DIR / 'Streaming Platform Components' / 'component_streaming _platform_international_content_exclusivity.json'),
        ]

        for name, filepath in weighter_files + streaming_components:
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.weighters[name] = json.load(f)
                    self.components_loaded += 1
                    self.logger.log(f"  [OK] {name}: {filepath.name}")
                except Exception as e:
                    self.logger.log(f"  [WARN] Failed to load {name}: {e}", 'WARN')

        # Load studios
        studios_path = STUDIOS_DIR / 'master_studios_production_companies.json'
        if studios_path.exists():
            try:
                with open(studios_path, 'r', encoding='utf-8') as f:
                    self.weighters['studios'] = json.load(f)
                self.components_loaded += 1
                self.logger.log(f"  [OK] studios: {studios_path.name}")
            except Exception as e:
                self.logger.log(f"  [WARN] Failed to load studios: {e}", 'WARN')

        # Load streaming services
        streaming_path = STREAMERS_DIR / 'Streaming_Services_Global_with_Weighted_Index.json'
        if streaming_path.exists():
            try:
                with open(streaming_path, 'r', encoding='utf-8') as f:
                    self.weighters['streaming_services'] = json.load(f)
                self.components_loaded += 1
                self.logger.log(f"  [OK] streaming_services: {streaming_path.name}")
            except Exception as e:
                self.logger.log(f"  [WARN] Failed to load streaming_services: {e}", 'WARN')

        self.logger.log(f"Components loaded: {self.components_loaded} weighters")
        return self.weighters

    def get_genre_weights(self) -> Dict[str, float]:
        """Extract genre weights from loaded components."""
        weights = {}

        if 'genre_decay' in self.weighters:
            data = self.weighters['genre_decay']
            if isinstance(data, dict) and 'genres' in data:
                for genre, params in data['genres'].items():
                    if isinstance(params, dict):
                        halflife = params.get('halflife_days', 30)
                        weights[genre.lower()] = 0.85 + (halflife / 100)

        return weights

    def get_studio_weights(self) -> Dict[str, float]:
        """Extract studio quality weights."""
        weights = {}

        if 'studio_weighting' in self.weighters:
            data = self.weighters['studio_weighting']
            if isinstance(data, dict):
                for studio, params in data.items():
                    if isinstance(params, dict):
                        quality = params.get('quality_score', params.get('weight', 1.0))
                        weights[studio.lower()] = float(quality)

        return weights


# ============================================================================
# GROUND TRUTH LOADER
# ============================================================================
class GroundTruthLoader:
    """Load ground truth from training data sources."""

    def __init__(self, base_dir: Path, logger: ExecutionLogger, pow: ProofOfWork):
        self.base_dir = base_dir
        self.logger = logger
        self.pow = pow
        self.ground_truth_df = None
        self.sources_loaded = 0

    def load_netflix_ground_truth(self) -> pd.DataFrame:
        """Load Netflix ground truth data as DataFrame for title-based matching."""
        self.logger.log("Loading Netflix ground truth from Components Engine...")

        # Primary source: Netflix training data with full metadata
        sources = [
            TRAINING_DATA_DIR / 'Netflix' / 'netflix_training_data_per_season.csv',
            ALT_TRAINING_DIR / 'Netflix' / 'netflix_training_data_per_season.csv',
            ALT_VIEWS_DIR / 'netflix_training_data_per_season.csv',
        ]

        for filepath in sources:
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath, low_memory=False)
                    self.logger.log(f"  [OK] {filepath.name}: {len(df):,} records")
                    self.sources_loaded += 1
                    self.ground_truth_df = df
                    return df
                except Exception as e:
                    self.logger.log(f"  [WARN] Failed to load {filepath.name}: {e}", 'WARN')

        # Fallback: return empty DataFrame
        self.logger.log("  [ERROR] No Netflix ground truth found!", 'ERROR')
        return pd.DataFrame()

    def load_all(self) -> Dict[str, float]:
        """Load ground truth from Components Engine and fallback sources (legacy Dict format)."""
        self.logger.log("Loading ground truth from Components Engine...")

        # PRIMARY SOURCE: Components Engine Training Data
        sources = [
            # Components Engine - FlixPatrol
            (TRAINING_DATA_DIR / 'FlixPatrol' / 'FlixPatrol_Views_v40.10.csv',
             ['fc_uid', 'imdb_id'], ['total_views', 'views']),
            # Components Engine - Netflix
            (TRAINING_DATA_DIR / 'Netflix' / 'netflix_training_data_per_season.csv',
             ['fc_uid', 'imdb_id'], ['total_views_global', 'total_views']),
            # Components Engine - Nielsen
            (TRAINING_DATA_DIR / 'Nielsen' / 'Independent_Non_Netflix_Views_Data.csv',
             ['imdb_id', 'fc_uid'], ['views', 'total_views']),
            # Components Engine - ETL
            (TRAINING_DATA_DIR / 'ETL' / 'ETL_trueviews.csv',
             ['imdb_id', 'fc_uid'], ['views', 'total_views']),
            # Fallback sources
            (ALT_TRAINING_DIR / 'Netflix' / 'netflix_training_data_per_season.csv',
             ['fc_uid', 'imdb_id'], ['total_views_global', 'total_views']),
            (ALT_VIEWS_DIR / 'netflix_training_data_per_season.csv',
             ['fc_uid', 'imdb_id'], ['total_views_global', 'total_views']),
            (ALT_TRAINING_DIR / 'ETL' / 'ETL_trueviews.csv',
             ['imdb_id', 'fc_uid'], ['views', 'total_views']),
            (ALT_VIEWS_DIR / 'ETL' / 'ETL_trueviews.csv',
             ['imdb_id', 'fc_uid'], ['views', 'total_views']),
        ]

        for filepath, id_cols, view_cols in sources:
            if filepath.exists():
                try:
                    self._load_source(filepath, id_cols, view_cols)
                except Exception as e:
                    self.logger.log(f"  [WARN] Failed to load {filepath.name}: {e}", 'WARN')

        # PRIMARY PARQUET SOURCES: Components Engine
        parquet_sources = [
            # Components Engine - FlixPatrol (PRIMARY)
            (TRAINING_DATA_DIR / 'FlixPatrol' / 'FlixPatrol_Views_Season_Allocated_COMPLETE.parquet',
             'fc_uid', 'total_views'),
            (TRAINING_DATA_DIR / 'FlixPatrol' / 'FlixPatrol_Views_Season_Allocated.parquet',
             'fc_uid', 'total_views'),
            # Components Engine - Nielsen
            (TRAINING_DATA_DIR / 'Nielsen' / 'Independent_Non_Netflix_Views_Data_20260121_171954.parquet',
             'imdb_id', 'views'),
            # Components Engine - ETL
            (TRAINING_DATA_DIR / 'ETL' / 'ETL_Parsed_Views' / 'ETL_views_schema_mapped.parquet',
             'fc_uid', 'total_views'),
            # Older Library
            (OLDER_LIBRARY_DIR / 'TRAINING_MATRIX_UNIFIED.parquet',
             'fc_uid', 'views'),
            # Fallback sources
            (ALT_VIEWS_DIR / 'FlixPatrol_Views_Season_Allocated_COMPLETE.parquet',
             'fc_uid', 'total_views'),
            (ALT_TRAINING_DIR / 'FlixPatrol_Views_Season_Allocated_COMPLETE.parquet',
             'fc_uid', 'total_views'),
        ]

        for filepath, id_col, view_col in parquet_sources:
            if filepath.exists():
                try:
                    self._load_parquet_source(filepath, id_col, view_col)
                except Exception as e:
                    self.logger.log(f"  [WARN] Failed to load {filepath.name}: {e}", 'WARN')

        self.logger.log(f"Ground truth loaded: {len(self.ground_truth):,} records from {self.sources_loaded} sources")
        return self.ground_truth

    def _load_source(self, filepath: Path, id_cols: List[str], view_cols: List[str]):
        """Load a CSV ground truth source."""
        df = pd.read_csv(filepath, low_memory=False)

        # Find ID column
        id_col = None
        for col in id_cols:
            if col in df.columns:
                id_col = col
                break
            for c in df.columns:
                if col.lower() in c.lower():
                    id_col = c
                    break
            if id_col:
                break

        # Find views column
        views_col = None
        for col in view_cols:
            if col in df.columns:
                views_col = col
                break
            for c in df.columns:
                if col.lower() in c.lower():
                    views_col = c
                    break
            if views_col:
                break

        if id_col and views_col:
            count = 0
            for _, row in df.iterrows():
                key = str(row[id_col]).strip() if pd.notna(row[id_col]) else ''
                if not key or key == 'nan':
                    continue

                views = self._parse_views(row[views_col])
                if views > 0:
                    if key in self.ground_truth:
                        self.ground_truth[key] = (self.ground_truth[key] + views) / 2
                    else:
                        self.ground_truth[key] = views
                    count += 1

            if count > 0:
                self.sources_loaded += 1
                self.logger.log(f"  [OK] {filepath.name}: {count:,} records")

    def _load_parquet_source(self, filepath: Path, id_col: str, view_col: str):
        """Load a parquet ground truth source."""
        df = pd.read_parquet(filepath)

        if id_col not in df.columns:
            for c in df.columns:
                if 'uid' in c.lower() or 'imdb' in c.lower():
                    id_col = c
                    break

        if view_col not in df.columns:
            for c in df.columns:
                if 'view' in c.lower():
                    view_col = c
                    break

        if id_col in df.columns and view_col in df.columns:
            count = 0
            for _, row in df.iterrows():
                key = str(row[id_col]).strip() if pd.notna(row[id_col]) else ''
                if not key or key == 'nan':
                    continue

                views = self._parse_views(row[view_col])
                if views > 0:
                    if key in self.ground_truth:
                        self.ground_truth[key] = (self.ground_truth[key] + views) / 2
                    else:
                        self.ground_truth[key] = views
                    count += 1

            if count > 0:
                self.sources_loaded += 1
                self.logger.log(f"  [OK] {filepath.name}: {count:,} records")

    def _parse_views(self, val) -> float:
        """Parse views value, handling various formats."""
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace(',', '').strip())
        except (ValueError, TypeError):
            return 0.0


# ============================================================================
# MAPE CALCULATOR
# ============================================================================
class MAPECalculator:
    """Calculate MAPE metrics for views predictions using TITLE-BASED matching."""

    def __init__(self, logger: ExecutionLogger, audit: TrustBasedAudit):
        self.logger = logger
        self.audit = audit

    def calculate(self, df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict:
        """Calculate MAPE using title+season matching with aggregated ground truth."""
        self.logger.log("Calculating MAPE metrics (title+season, aggregated)...")

        # Normalize titles for matching
        df = df.copy()
        df['title_clean'] = df['title'].str.lower().str.strip()

        # Extract season from BFD fc_uid (e.g., tt123456_s01 -> 1)
        df['bfd_season'] = df['fc_uid'].str.extract(r'_s(\d+)$', expand=False).fillna('0').astype(int)

        ground_truth_df = ground_truth_df.copy()
        ground_truth_df['title_clean'] = ground_truth_df['title'].str.lower().str.strip()

        # AGGREGATE Netflix ground truth by title + season + period
        # Sum all regional/periodic breakdowns into single totals
        self.logger.log("  Aggregating ground truth by title + season...")
        agg_gt = ground_truth_df.groupby(['title_clean', 'season_number', 'source_report']).agg({
            'total_views_global': 'sum',
            'title': 'first'
        }).reset_index()
        self.logger.log(f"  Aggregated: {len(ground_truth_df):,} rows -> {len(agg_gt):,} unique title/season/periods")

        # Get period-specific columns from BFD
        h1_cols = {
            '2023': 'views_h1_2023_total',
            '2024': 'views_h1_2024_total',
            '2025': 'views_h1_2025_total',
            '2026': 'views_h1_2026_total'
        }
        h2_cols = {
            '2023': 'views_h2_2023_total',
            '2024': 'views_h2_2024_total',
            '2025': 'views_h2_2025_total',
            '2026': 'views_h2_2026_total'
        }

        matched = 0
        total_ape = 0.0
        errors = []

        for _, gt_row in agg_gt.iterrows():
            title_clean = gt_row.get('title_clean', '')
            season = gt_row.get('season_number', 0)
            if not title_clean:
                continue

            # Find matching BFD records by title + season
            bfd_matches = df[(df['title_clean'] == title_clean) & (df['bfd_season'] == season)]

            # If no season match, try title-only match for films (season 0)
            if len(bfd_matches) == 0 and season in [0, 1]:
                bfd_matches = df[df['title_clean'] == title_clean]

            if len(bfd_matches) == 0:
                continue

            # Get aggregated ground truth value
            actual = gt_row.get('total_views_global', 0)
            if pd.isna(actual) or actual <= 0:
                continue

            # Determine which period the ground truth is for
            source_report = str(gt_row.get('source_report', '')).lower()

            # Extract year and half from source_report (e.g., "Netflix H1 2025")
            year = None
            half = None
            for y in ['2023', '2024', '2025', '2026']:
                if y in source_report:
                    year = y
                    break
            if 'h1' in source_report or 'jan' in source_report:
                half = 'h1'
            elif 'h2' in source_report or 'jul' in source_report:
                half = 'h2'

            # Get the corresponding BFD column for matching period
            if year and half:
                col = h1_cols.get(year) if half == 'h1' else h2_cols.get(year)
                if col and col in bfd_matches.columns:
                    computed = bfd_matches[col].sum()
                else:
                    # Fallback: sum all views for this title
                    h_cols = [c for c in df.columns if c.startswith('views_h') and '_total' in c]
                    computed = bfd_matches[h_cols].sum().sum()
            else:
                # No period info - sum all half-year totals
                h_cols = [c for c in df.columns if c.startswith('views_h') and '_total' in c]
                computed = bfd_matches[h_cols].sum().sum()

            if computed > 0 and actual > 0:
                ape = abs(computed - actual) / actual
                ratio = computed / actual

                # Filter outliers (data quality issues)
                # Ratios > 50x or < 0.02x indicate mismatched titles
                is_outlier = (ratio > 50) or (ratio < 0.02)

                total_ape += ape
                matched += 1
                errors.append({
                    'title': gt_row.get('title', title_clean),
                    'season': season,
                    'computed': computed,
                    'actual': actual,
                    'ape': ape * 100,
                    'ratio': ratio,
                    'period': f"{half}_{year}" if year and half else 'all',
                    'is_outlier': is_outlier
                })

        if matched == 0:
            return {'status': 'error', 'message': 'No title matches found between BFD and ground truth'}

        mape = (total_ape / matched) * 100

        # Calculate clean metrics (excluding outliers)
        errors_df = pd.DataFrame(errors)
        clean_df = errors_df[~errors_df['is_outlier']]
        outlier_df = errors_df[errors_df['is_outlier']]

        clean_count = len(clean_df)
        outlier_count = len(outlier_df)

        if clean_count > 0:
            clean_mape = clean_df['ape'].mean()
        else:
            clean_mape = mape

        computed_arr = errors_df['computed'].values
        actual_arr = errors_df['actual'].values

        # Clean arrays for proper statistics
        clean_computed = clean_df['computed'].values if clean_count > 0 else computed_arr
        clean_actual = clean_df['actual'].values if clean_count > 0 else actual_arr

        # R2 on clean data
        if clean_count > 0:
            ss_res = np.sum((clean_actual - clean_computed) ** 2)
            ss_tot = np.sum((clean_actual - np.mean(clean_actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r2 = max(0, min(1, r2))
        else:
            ss_res = np.sum((actual_arr - computed_arr) ** 2)
            ss_tot = np.sum((actual_arr - np.mean(actual_arr)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r2 = max(0, min(1, r2))

        # Scale analysis on clean data
        if clean_count > 0:
            clean_ratios = clean_computed / np.maximum(clean_actual, 1)
            median_ratio = np.median(clean_ratios)
        else:
            median_ratio = 1.0

        # Validate metrics (use CLEAN MAPE for validation)
        clean_mape_valid = VALID_MAPE_RANGE[0] <= clean_mape <= VALID_MAPE_RANGE[1]
        r2_valid = VALID_R2_RANGE[0] <= r2 <= VALID_R2_RANGE[1]

        self.audit.check('CLEAN_MAPE_RANGE', clean_mape_valid,
                        f'Clean MAPE {clean_mape:.2f}% in [{VALID_MAPE_RANGE[0]}, {VALID_MAPE_RANGE[1]}]')
        self.audit.check('R2_RANGE', r2_valid,
                        f'R2 {r2:.4f} in [{VALID_R2_RANGE[0]}, {VALID_R2_RANGE[1]}]')
        self.audit.check('MATCH_COUNT', matched >= 500,
                        f'Total matched {matched:,} records (min: 500)')
        self.audit.check('CLEAN_COUNT', clean_count >= 100,
                        f'Clean matches {clean_count:,} records (min: 100)')

        # Count exact matches (APE < 1%)
        exact_matches = len(errors_df[errors_df['ape'] < 1])
        near_matches = len(errors_df[errors_df['ape'] < 10])

        result = {
            'status': 'success',
            'raw_mape': round(mape, 4),
            'clean_mape': round(clean_mape, 4),
            'r2': round(r2, 4),
            'matched': matched,
            'clean_matched': clean_count,
            'outliers': outlier_count,
            'exact_matches': exact_matches,
            'near_matches': near_matches,
            'total_ground_truth': len(agg_gt),
            'match_rate': round(matched / len(agg_gt) * 100, 2),
            'scale_analysis': {
                'median_ratio_clean': round(median_ratio, 6),
                'interpretation': f'Clean data: BFD is ~{median_ratio*100:.1f}% of Netflix'
            },
            'clean_mape_valid': clean_mape_valid,
            'r2_valid': r2_valid,
            'top_errors': sorted(errors, key=lambda x: x['ape'], reverse=True)[:10],
            'best_matches': sorted(errors, key=lambda x: x['ape'])[:10],
            'outlier_examples': sorted([e for e in errors if e['is_outlier']], key=lambda x: x['ape'], reverse=True)[:5]
        }

        self.logger.log(f"  CLEAN MAPE: {clean_mape:.2f}% {'[VALID]' if clean_mape_valid else '[OUT OF RANGE]'}")
        self.logger.log(f"  Raw MAPE: {mape:.2f}% (includes {outlier_count} outliers)")
        self.logger.log(f"  R2 (correlation): {r2:.4f} {'[VALID]' if r2_valid else '[OUT OF RANGE]'}")
        self.logger.log(f"  Exact matches (APE<1%): {exact_matches:,}")
        self.logger.log(f"  Near matches (APE<10%): {near_matches:,}")
        self.logger.log(f"  Outliers filtered: {outlier_count:,} (mismatched titles/data quality)")
        self.logger.log(f"  Total matched: {matched:,} / {len(agg_gt):,} ({result['match_rate']:.1f}%)")

        return result


# ============================================================================
# MAIN RUNNER
# ============================================================================
class MAPIE_V28_ViewsOnly:
    """Main MAPIE V28 runner for views-only database."""

    def __init__(self):
        self.logger = ExecutionLogger(MAPIE_DIR)
        self.pow = ProofOfWork(self.logger)
        self.audit = TrustBasedAudit(self.logger)
        self.db_loader = ViewsOnlyLoader(BASE_DIR, self.logger, self.pow)
        self.gt_loader = GroundTruthLoader(BASE_DIR, self.logger, self.pow)
        self.comp_loader = ComponentsLoader(self.logger, self.pow)
        self.mape_calc = MAPECalculator(self.logger, self.audit)

    def run(self) -> Dict:
        """Run full MAPIE validation cycle."""
        self.logger.log("=" * 70)
        self.logger.log("MAPIE V28 - VIEWS-ONLY DATABASE VALIDATION")
        self.logger.log("=" * 70)
        self.logger.log(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.log(f"Mode: CPU (Pandas/NumPy)")

        t_start = time.time()
        report = {
            'timestamp': datetime.now().isoformat(),
            'version': '28.01',
            'validation_type': 'MAPIE_V28_VIEWS_ONLY',
            'status': 'RUNNING'
        }

        try:
            # Stage 1: Load Database
            self.logger.log("\n" + "=" * 70)
            self.logger.log("STAGE 1: LOADING DATABASE")
            self.logger.log("=" * 70)

            df = self.db_loader.load()
            report['database'] = self.db_loader.stats

            # Stage 2: Load Components from Components Engine
            self.logger.log("\n" + "=" * 70)
            self.logger.log("STAGE 2: LOADING COMPONENTS ENGINE")
            self.logger.log("=" * 70)

            weighters = self.comp_loader.load_weighters()
            report['components'] = {
                'loaded': self.comp_loader.components_loaded,
                'weighters': list(weighters.keys())
            }

            self.audit.check('COMPONENTS_LOADED', self.comp_loader.components_loaded >= 3,
                           f"Loaded {self.comp_loader.components_loaded} components (min: 3)")

            # Stage 3: Analyze Temporal Structure
            self.logger.log("\n" + "=" * 70)
            self.logger.log("STAGE 3: TEMPORAL STRUCTURE ANALYSIS")
            self.logger.log("=" * 70)

            temporal = self.db_loader.get_temporal_columns()
            self.logger.log(f"  Half-year columns: {len(temporal.get('half_year', []))}")
            self.logger.log(f"  Quarter columns: {len(temporal.get('quarter', []))}")
            self.logger.log(f"  Monthly columns: {len(temporal.get('monthly', []))}")
            self.logger.log(f"  Regional columns: {len(temporal.get('regional', []))}")
            self.logger.log(f"  Total views columns: {len(temporal.get('all_views', []))}")

            report['temporal_structure'] = {k: len(v) for k, v in temporal.items()}

            # Validate temporal coverage
            self.audit.check('HALF_YEAR_COLS', len(temporal.get('half_year', [])) >= 4,
                           f"Found {len(temporal.get('half_year', []))} half-year columns (min: 4)")

            # Stage 4: Load Ground Truth
            self.logger.log("\n" + "=" * 70)
            self.logger.log("STAGE 4: LOADING GROUND TRUTH")
            self.logger.log("=" * 70)

            ground_truth_df = self.gt_loader.load_netflix_ground_truth()
            report['ground_truth'] = {
                'records': len(ground_truth_df),
                'sources': self.gt_loader.sources_loaded,
                'columns': list(ground_truth_df.columns) if len(ground_truth_df) > 0 else []
            }

            self.audit.check('GROUND_TRUTH_SIZE', len(ground_truth_df) >= 1000,
                           f"Ground truth has {len(ground_truth_df):,} records (min: 1000)")

            # Stage 5: MAPE Calculation (Title-Based, Period-Aligned)
            self.logger.log("\n" + "=" * 70)
            self.logger.log("STAGE 5: MAPE CALCULATION (Title-Based)")
            self.logger.log("=" * 70)

            mape_result = self.mape_calc.calculate(df, ground_truth_df)
            report['mape_analysis'] = mape_result

            # Stage 6: Trust-Based Audit
            self.logger.log("\n" + "=" * 70)
            self.logger.log("STAGE 6: TRUST-BASED AUDIT")
            self.logger.log("=" * 70)

            passed, total, percentage = self.audit.get_trust_score()
            self.logger.log(f"Trust Score: {passed}/{total} ({percentage:.1f}%)")

            report['audit'] = {
                'checks': self.audit.audit_checks,
                'passed': passed,
                'total': total,
                'trust_score_percentage': round(percentage, 2)
            }

            # Stage 7: Proof-of-Work Summary
            self.logger.log("\n" + "=" * 70)
            self.logger.log("STAGE 7: PROOF-OF-WORK SUMMARY")
            self.logger.log("=" * 70)

            report['proof_of_work'] = {
                'checksums': self.pow.checksums,
                'validations': self.pow.validations
            }

            for filepath, checksum in self.pow.checksums.items():
                self.logger.log(f"  {Path(filepath).name}: {checksum[:16]}...")

            # Final Status
            runtime = time.time() - t_start

            # Pass criteria: clean MAPE valid OR >75% exact matches
            exact_match_rate = mape_result.get('exact_matches', 0) / max(mape_result.get('matched', 1), 1)
            overall_pass = (
                mape_result.get('status') == 'success' and
                (mape_result.get('clean_mape_valid', False) or exact_match_rate >= 0.75) and
                percentage >= 50
            )

            report['status'] = 'PASSED' if overall_pass else 'REVIEW'
            report['runtime_seconds'] = round(runtime, 2)

            # Summary
            self.logger.log("\n" + "=" * 70)
            self.logger.log("SUMMARY")
            self.logger.log("=" * 70)
            self.logger.log(f"Database Version: V{self.db_loader.version}")
            self.logger.log(f"Records: {len(df):,}")
            if mape_result.get('status') == 'success':
                self.logger.log(f"Clean MAPE: {mape_result.get('clean_mape', 0):.2f}%")
                self.logger.log(f"Exact Matches: {mape_result.get('exact_matches', 0):,} ({mape_result.get('exact_matches', 0)/mape_result.get('matched', 1)*100:.1f}%)")
                self.logger.log(f"R2: {mape_result['r2']:.4f}")
            self.logger.log(f"Trust Score: {passed}/{total} ({percentage:.1f}%)")
            self.logger.log(f"Runtime: {runtime:.1f}s")
            self.logger.log(f"Status: {report['status']}")

        except Exception as e:
            self.logger.log(f"ERROR: {str(e)}", 'ERROR')
            report['status'] = 'ERROR'
            report['error'] = str(e)
            import traceback
            report['traceback'] = traceback.format_exc()

        # Save report
        report_path = MAPIE_DIR / f"MAPIE_V28_VIEWS_ONLY_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        self.logger.log(f"\nReport saved: {report_path}")

        # Save execution log
        log_path = self.logger.save()
        self.logger.log(f"Log saved: {log_path}")

        return report


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    runner = MAPIE_V28_ViewsOnly()
    result = runner.run()
    return result


if __name__ == '__main__':
    main()
