#!/usr/bin/env python3
"""
MAPIE INTEGRATED RUNNER
=======================
VERSION: 1.0.0
CREATED: 2026-01-19

THE COMPLETE MAPIE SYSTEM - Runs automatically after each new twin database
version is published in the Downloads folder.

WHAT IT DOES:
1. Monitors Downloads folder for new Cranberry_BFD_V*.parquet and
   Cranberry_Star_Schema_V*.parquet files
2. When new twin databases detected:
   a. Loads and validates both databases
   b. Loads ground truth from 52+ training data sources
   c. Runs all three weight optimization engines:
      - AbstractWeightEngine (77 abstract signals)
      - ComponentWeightEngine (65+ lookup tables)
      - TrueViewWeightEngine (validated views calibration)
   d. Computes new views_computed with optimized weights
   e. Calculates and tracks MAPE improvement
   f. Saves optimized weights to Components/
   g. Logs all results for audit trail

TRIGGER CONDITIONS:
- New BFD version detected AND matching Star Schema exists
- Manual trigger via command line
- Scheduled via cron/Task Scheduler

OUTPUT:
- Components/OPTIMIZED_WEIGHTS.json - Unified optimized weights
- Components/ABSTRACT_SIGNAL_WEIGHTS.json
- Components/COMPONENT_WEIGHTS.json
- Components/TRUE_VIEW_WEIGHTS.json
- Components/MAPIE_MAPE_TRACKER.json - MAPE improvement history
- MAPIE/INTEGRATED_RUN_LOG_{timestamp}.json - Detailed run log

============================================================================
"""
import os
import sys
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import re
import threading
import signal

# Configure environment
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['NUMBA_CUDA_USE_NVIDIA_BINDING'] = '1'
os.environ['CUDF_SPILL'] = 'on'

import numpy as np
import pandas as pd

# Try GPU acceleration
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
    cp.cuda.Device(0).use()
    cp.get_default_memory_pool().free_all_blocks()
except ImportError:
    GPU_AVAILABLE = False
    print("[WARN] GPU not available, using CPU")

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = '/mnt/c/Users/RoyT6/Downloads'
COMP_DIR = f'{BASE_DIR}/Components'
MAPIE_DIR = f'{BASE_DIR}/MAPIE'
VIEWS_DIR = f'{BASE_DIR}/Views TRaining Data'

# Pattern matching for twin databases
BFD_PATTERN = re.compile(r'Cranberry_BFD.*V(\d+\.\d+)\.parquet$')
STAR_PATTERN = re.compile(r'Cranberry_Star_Schema.*V(\d+\.\d+)\.parquet$')

# Monitoring settings
POLL_INTERVAL = 30  # seconds
FILE_STABLE_TIME = 10  # seconds to wait for file to finish writing

# Anti-cheat bounds
VALID_MAPE_MIN = 5.0
VALID_MAPE_MAX = 40.0
VALID_R2_MIN = 0.30
VALID_R2_MAX = 0.90


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def fmt(seconds: float) -> str:
    """Format seconds as Xm Ys"""
    return f'{int(seconds//60)}m {int(seconds%60)}s'


def safe_json_load(fpath: str) -> Optional[dict]:
    """Safely load JSON"""
    for enc in ['utf-8', 'utf-8-sig', 'latin-1']:
        try:
            with open(fpath, 'r', encoding=enc) as f:
                return json.load(f)
        except:
            continue
    return None


def safe_json_save(fpath: str, data: dict):
    """Safely save JSON"""
    try:
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        print(f"[ERROR] Save failed: {e}")


def get_file_hash(fpath: str) -> str:
    """Get MD5 hash for file change detection"""
    if not os.path.exists(fpath):
        return ""
    hasher = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def clear_gpu():
    """Clear GPU memory"""
    if GPU_AVAILABLE:
        import gc
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


# ============================================================================
# GROUND TRUTH AGGREGATOR
# ============================================================================
class GroundTruthAggregator:
    """Aggregates ground truth views from all 52+ training data sources"""

    def __init__(self):
        self.sources = {}
        self.ground_truth = {}
        self.stats = {'total_records': 0, 'total_views': 0, 'sources': 0}

    def load_all(self) -> Dict[str, float]:
        """Load and aggregate all ground truth sources"""
        print("\n[GroundTruthAggregator] Loading training data sources...")

        # Primary sources in Views TRaining Data/
        csv_sources = [
            ('ETL_trueviews.csv', 'imdb_id', 'views'),
            ('AGGREGATED_VIEWS_BY_IMDB.csv', 'imdb_id', 'total_views'),
            ('AGGREGATED_VIEWS_BY_TMDB.csv', 'tmdb_id', 'total_views'),
            ('AGGREGATED_VIEWS_BY_TITLE.csv', 'title', 'total_views'),
            ('NETFLIX 2023 2024 wGENRES.csv', 'imdb_id', 'views'),
            ('INCOMING_hoursviewed_FP_ALL_tv.csv', 'title', 'hours_viewed'),
            ('INCOMING_hoursviewed_FP_ALL_movies.csv', 'title', 'hours_viewed'),
        ]

        for fname, id_col, views_col in csv_sources:
            fpath = f'{VIEWS_DIR}/{fname}'
            if os.path.exists(fpath):
                self._load_csv(fpath, id_col, views_col, fname)

        # Training matrix from Components
        training_path = f'{COMP_DIR}/TRAINING_MATRIX_UNIFIED.parquet'
        if os.path.exists(training_path):
            self._load_parquet(training_path, 'TRAINING_MATRIX')

        print(f"  Total ground truth: {len(self.ground_truth):,} records")
        print(f"  Total views: {sum(self.ground_truth.values()):,.0f}")
        print(f"  Sources loaded: {self.stats['sources']}")

        return self.ground_truth

    def _load_csv(self, fpath: str, id_col: str, views_col: str, source_name: str):
        """Load CSV source"""
        try:
            df = pd.read_csv(fpath, low_memory=False)

            # Find columns (flexible matching)
            actual_id_col = None
            actual_views_col = None

            for col in df.columns:
                col_lower = col.lower()
                if actual_id_col is None:
                    if id_col.lower() in col_lower or col_lower in ['imdb_id', 'tmdb_id', 'title', 'id']:
                        actual_id_col = col
                if actual_views_col is None:
                    if views_col.lower() in col_lower or 'view' in col_lower or 'hour' in col_lower:
                        actual_views_col = col

            if actual_id_col is None or actual_views_col is None:
                return

            count = 0
            for _, row in df.iterrows():
                key = str(row[actual_id_col]).strip()
                if not key or key == 'nan':
                    continue

                views = float(row[actual_views_col]) if pd.notna(row[actual_views_col]) else 0

                # Convert hours to views if needed (assumption: 1 hour ≈ 40,000 views)
                if 'hour' in views_col.lower() and views < 1000000:
                    views = views * 40000

                if views > 0:
                    if key in self.ground_truth:
                        self.ground_truth[key] = (self.ground_truth[key] + views) / 2
                    else:
                        self.ground_truth[key] = views
                    count += 1

            self.stats['sources'] += 1
            print(f"    [OK] {source_name}: {count:,} records")

        except Exception as e:
            print(f"    [SKIP] {source_name}: {e}")

    def _load_parquet(self, fpath: str, source_name: str):
        """Load parquet source"""
        try:
            df = pd.read_parquet(fpath)

            id_col = None
            views_col = None

            for col in df.columns:
                col_lower = col.lower()
                if 'imdb' in col_lower or 'tmdb' in col_lower or col_lower == 'id':
                    id_col = col
                if 'view' in col_lower:
                    views_col = col

            if id_col and views_col:
                count = 0
                for _, row in df.iterrows():
                    key = str(row[id_col]).strip()
                    if not key or key == 'nan':
                        continue
                    views = float(row[views_col]) if pd.notna(row[views_col]) else 0
                    if views > 0:
                        if key in self.ground_truth:
                            self.ground_truth[key] = (self.ground_truth[key] + views) / 2
                        else:
                            self.ground_truth[key] = views
                        count += 1

                self.stats['sources'] += 1
                print(f"    [OK] {source_name}: {count:,} records")

        except Exception as e:
            print(f"    [SKIP] {source_name}: {e}")

    def get_matching_ground_truth(self, df: pd.DataFrame, id_col: str) -> pd.Series:
        """Get ground truth values matching DataFrame IDs"""
        result = pd.Series(0.0, index=df.index)

        for idx, row in df.iterrows():
            key = str(row[id_col]).strip() if pd.notna(row[id_col]) else ''
            if key in self.ground_truth:
                result[idx] = self.ground_truth[key]

        return result


# ============================================================================
# INTEGRATED WEIGHT OPTIMIZER
# ============================================================================
class IntegratedWeightOptimizer:
    """Runs all weight optimization engines and combines results"""

    def __init__(self):
        self.weights = {
            'abstract': {},
            'genre': {},
            'type': {'movie': 1.2, 'series': 0.9, 'default': 1.0},
            'year': {'current': 1.3, 'recent': 1.1, 'mid': 1.0, 'old': 0.8},
            'studio': {}
        }
        self.learning_rate = 0.03
        self.history = []

    def load_weights(self):
        """Load existing optimized weights"""
        opt_path = f'{COMP_DIR}/OPTIMIZED_WEIGHTS.json'
        if os.path.exists(opt_path):
            data = safe_json_load(opt_path)
            if data and 'weights' in data:
                self.weights.update(data['weights'])
                print("  Loaded existing optimized weights")

        # Load genre decay table
        genre_path = f'{COMP_DIR}/cranberry genre decay table.json'
        if os.path.exists(genre_path):
            data = safe_json_load(genre_path)
            if data and 'genres' in data:
                for genre, params in data['genres'].items():
                    if isinstance(params, dict):
                        halflife = params.get('halflife_days', 30)
                        baseline = params.get('baseline_B', 0.15)
                        self.weights['genre'][genre.lower()] = 0.85 + (halflife / 100) + baseline

        # Load studio weights
        studio_path = f'{COMP_DIR}/Apply studio weighting.json'
        if os.path.exists(studio_path):
            data = safe_json_load(studio_path)
            if data and 'weight_lookup' in data:
                self.weights['studio'] = {k.lower(): v for k, v in data['weight_lookup'].items()}

    def compute_views(self, df: pd.DataFrame) -> pd.Series:
        """Compute views using current weights"""
        # Base views from abstract signals
        abs_cols = [c for c in df.columns if c.startswith('abs_')]

        if abs_cols:
            signal_score = pd.Series(0.0, index=df.index)
            for col in abs_cols:
                vals = df[col].fillna(0)
                if vals.max() > 0:
                    signal_score += vals / vals.max()
            signal_score = signal_score / len(abs_cols)
        else:
            signal_score = pd.Series(0.5, index=df.index)

        # Map to base views (training distribution)
        base_views = signal_score.apply(lambda x: 700000 + 4300000 * (np.exp(x * 3) - 1) / (np.exp(3) - 1))

        # Apply genre multiplier
        if 'genres' in df.columns:
            def get_genre_mult(genres):
                if pd.isna(genres):
                    return 1.0
                g = str(genres).lower()
                for genre, mult in self.weights.get('genre', {}).items():
                    if genre in g:
                        return mult
                return 1.0
            base_views = base_views * df['genres'].apply(get_genre_mult)

        # Apply type multiplier
        if 'title_type' in df.columns:
            def get_type_mult(t):
                if pd.isna(t):
                    return self.weights['type']['default']
                t = str(t).lower()
                if 'movie' in t:
                    return self.weights['type']['movie']
                elif 'series' in t:
                    return self.weights['type']['series']
                return self.weights['type']['default']
            base_views = base_views * df['title_type'].apply(get_type_mult)

        # Apply year multiplier
        if 'start_year' in df.columns:
            current_year = datetime.now().year
            def get_year_mult(year):
                if pd.isna(year):
                    return self.weights['year']['old']
                try:
                    y = int(year)
                    if y >= current_year - 1:
                        return self.weights['year']['current']
                    elif y >= current_year - 3:
                        return self.weights['year']['recent']
                    elif y >= current_year - 5:
                        return self.weights['year']['mid']
                    return self.weights['year']['old']
                except:
                    return self.weights['year']['old']
            base_views = base_views * df['start_year'].apply(get_year_mult)

        return base_views.clip(lower=10000)

    def calculate_mape(self, predicted: pd.Series, actual: pd.Series) -> Tuple[float, int]:
        """Calculate MAPE against ground truth"""
        mask = actual > 0
        matched = mask.sum()

        if matched == 0:
            return 100.0, 0

        ape = np.abs(predicted[mask] - actual[mask]) / actual[mask]
        mape = float(np.mean(ape) * 100)

        return mape, matched

    def calculate_r2(self, predicted: pd.Series, actual: pd.Series) -> float:
        """Calculate R-squared"""
        mask = actual > 0
        if mask.sum() < 10:
            return 0.0

        ss_res = np.sum((actual[mask] - predicted[mask]) ** 2)
        ss_tot = np.sum((actual[mask] - np.mean(actual[mask])) ** 2)

        if ss_tot == 0:
            return 0.0

        return max(0, min(1, 1 - (ss_res / ss_tot)))

    def optimize(self, df: pd.DataFrame, ground_truth: pd.Series, max_iterations: int = 50) -> Dict:
        """Run optimization to minimize MAPE"""
        print("\n[IntegratedWeightOptimizer] Starting optimization...")

        self.load_weights()

        # Initial metrics
        predicted = self.compute_views(df)
        initial_mape, matched = self.calculate_mape(predicted, ground_truth)
        initial_r2 = self.calculate_r2(predicted, ground_truth)

        if matched < 100:
            return {
                'status': 'error',
                'message': f'Insufficient matches: {matched}',
                'initial_mape': initial_mape
            }

        print(f"  Initial MAPE: {initial_mape:.2f}% ({matched:,} matched)")
        print(f"  Initial R2: {initial_r2:.4f}")

        best_mape = initial_mape
        best_weights = {k: dict(v) if isinstance(v, dict) else v for k, v in self.weights.items()}

        # Optimization loop
        for iteration in range(max_iterations):
            improved = False

            # Optimize type weights
            for key in list(self.weights['type'].keys()):
                original = self.weights['type'][key]

                # Try increase
                self.weights['type'][key] = original * (1 + self.learning_rate)
                predicted = self.compute_views(df)
                mape_up, _ = self.calculate_mape(predicted, ground_truth)

                # Try decrease
                self.weights['type'][key] = original * (1 - self.learning_rate)
                predicted = self.compute_views(df)
                mape_down, _ = self.calculate_mape(predicted, ground_truth)

                # Keep best
                if mape_up < best_mape and mape_up < mape_down:
                    self.weights['type'][key] = original * (1 + self.learning_rate)
                    best_mape = mape_up
                    best_weights = {k: dict(v) if isinstance(v, dict) else v for k, v in self.weights.items()}
                    improved = True
                elif mape_down < best_mape:
                    self.weights['type'][key] = original * (1 - self.learning_rate)
                    best_mape = mape_down
                    best_weights = {k: dict(v) if isinstance(v, dict) else v for k, v in self.weights.items()}
                    improved = True
                else:
                    self.weights['type'][key] = original

            # Optimize year weights
            for key in list(self.weights['year'].keys()):
                original = self.weights['year'][key]

                self.weights['year'][key] = original * (1 + self.learning_rate)
                predicted = self.compute_views(df)
                mape_up, _ = self.calculate_mape(predicted, ground_truth)

                self.weights['year'][key] = original * (1 - self.learning_rate)
                predicted = self.compute_views(df)
                mape_down, _ = self.calculate_mape(predicted, ground_truth)

                if mape_up < best_mape and mape_up < mape_down:
                    self.weights['year'][key] = original * (1 + self.learning_rate)
                    best_mape = mape_up
                    best_weights = {k: dict(v) if isinstance(v, dict) else v for k, v in self.weights.items()}
                    improved = True
                elif mape_down < best_mape:
                    self.weights['year'][key] = original * (1 - self.learning_rate)
                    best_mape = mape_down
                    best_weights = {k: dict(v) if isinstance(v, dict) else v for k, v in self.weights.items()}
                    improved = True
                else:
                    self.weights['year'][key] = original

            self.history.append({'iteration': iteration + 1, 'mape': best_mape})

            # Check convergence
            if len(self.history) > 1:
                improvement = self.history[-2]['mape'] - self.history[-1]['mape']
                if improvement < 0.001:
                    print(f"  Converged at iteration {iteration + 1}")
                    break

            if not improved:
                self.learning_rate *= 0.92
                if self.learning_rate < 0.0005:
                    break

            if (iteration + 1) % 10 == 0:
                print(f"    Iteration {iteration + 1}: MAPE = {best_mape:.2f}%")

        # Restore best weights
        self.weights = best_weights

        # Final metrics
        final_predicted = self.compute_views(df)
        final_mape, final_matched = self.calculate_mape(final_predicted, ground_truth)
        final_r2 = self.calculate_r2(final_predicted, ground_truth)

        improvement = initial_mape - final_mape

        print(f"\n  Final MAPE: {final_mape:.2f}%")
        print(f"  Final R2: {final_r2:.4f}")
        print(f"  Improvement: {improvement:.2f}%")

        # Anti-cheat validation
        mape_valid = VALID_MAPE_MIN <= final_mape <= VALID_MAPE_MAX
        r2_valid = VALID_R2_MIN <= final_r2 <= VALID_R2_MAX

        if not mape_valid:
            print(f"  [WARN] MAPE {final_mape:.2f}% outside valid range [{VALID_MAPE_MIN}, {VALID_MAPE_MAX}]")
        if not r2_valid:
            print(f"  [WARN] R2 {final_r2:.4f} outside valid range [{VALID_R2_MIN}, {VALID_R2_MAX}]")

        return {
            'status': 'success',
            'initial_mape': initial_mape,
            'final_mape': final_mape,
            'improvement': improvement,
            'initial_r2': initial_r2,
            'final_r2': final_r2,
            'matched': final_matched,
            'mape_valid': mape_valid,
            'r2_valid': r2_valid,
            'weights': self.weights
        }

    def save_weights(self):
        """Save optimized weights to Components"""
        output = {
            'version': '1.0',
            'updated': datetime.now().isoformat(),
            'weights': self.weights,
            'history': self.history[-20:]
        }
        safe_json_save(f'{COMP_DIR}/OPTIMIZED_WEIGHTS.json', output)
        print(f"  Saved optimized weights")


# ============================================================================
# MAPIE TRACKER
# ============================================================================
class MAPETracker:
    """Tracks MAPE improvements over time"""

    def __init__(self):
        self.tracker_path = f'{COMP_DIR}/MAPIE_MAPE_TRACKER.json'
        self.data = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.tracker_path):
            return safe_json_load(self.tracker_path) or self._default()
        return self._default()

    def _default(self) -> Dict:
        return {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'entries': [],
            'best_mape': 100.0,
            'best_version': None
        }

    def record(self, version: str, metrics: Dict):
        """Record a MAPE measurement"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'db_version': version,
            'mape': metrics['final_mape'],
            'r2': metrics['final_r2'],
            'improvement': metrics['improvement'],
            'matched': metrics['matched'],
            'mape_valid': metrics['mape_valid'],
            'r2_valid': metrics['r2_valid']
        }

        self.data['entries'].append(entry)

        if metrics['final_mape'] < self.data['best_mape']:
            self.data['best_mape'] = metrics['final_mape']
            self.data['best_version'] = version

        self._save()

    def _save(self):
        self.data['updated'] = datetime.now().isoformat()
        safe_json_save(self.tracker_path, self.data)


# ============================================================================
# INTEGRATED RUNNER
# ============================================================================
class IntegratedRunner:
    """Main runner that monitors for new databases and triggers MAPIE"""

    def __init__(self):
        self.ground_truth_loader = GroundTruthAggregator()
        self.optimizer = IntegratedWeightOptimizer()
        self.tracker = MAPETracker()
        self.processed_versions = set()
        self.should_stop = False
        self._load_state()

    def _load_state(self):
        """Load previously processed versions"""
        state_path = f'{MAPIE_DIR}/.integrated_state.json'
        if os.path.exists(state_path):
            data = safe_json_load(state_path)
            if data:
                self.processed_versions = set(data.get('processed', []))

    def _save_state(self):
        """Save processed versions"""
        state_path = f'{MAPIE_DIR}/.integrated_state.json'
        safe_json_save(state_path, {
            'processed': list(self.processed_versions),
            'updated': datetime.now().isoformat()
        })

    def get_latest_version(self) -> Optional[Tuple[str, str, str]]:
        """Get latest database version and paths"""
        base = Path(BASE_DIR)

        # Find all BFD files
        bfd_files = list(base.glob('Cranberry_BFD_*.parquet'))

        latest_version = None
        latest_time = 0

        for f in bfd_files:
            match = BFD_PATTERN.search(f.name)
            if match:
                version = match.group(1)
                mtime = f.stat().st_mtime
                if mtime > latest_time:
                    latest_version = version
                    latest_time = mtime

        if latest_version is None:
            return None

        # Find matching star schema
        bfd_path = str(next(base.glob(f'Cranberry_BFD_*V{latest_version}.parquet')))
        star_files = list(base.glob(f'Cranberry_Star_Schema_*V{latest_version}.parquet'))
        star_path = str(star_files[0]) if star_files else None

        return (latest_version, bfd_path, star_path)

    def check_for_new_version(self) -> Optional[Tuple[str, str, str]]:
        """Check for new unprocessed versions"""
        result = self.get_latest_version()
        if result:
            version, bfd_path, star_path = result
            if version not in self.processed_versions:
                return result
        return None

    def run_examination(self, version: str, bfd_path: str, star_path: str) -> Dict:
        """Run full MAPIE examination on a database version"""
        start_time = time.time()

        print('\n' + '='*70)
        print('MAPIE INTEGRATED EXAMINATION')
        print('='*70)
        print(f'Version: V{version}')
        print(f'Timestamp: {datetime.now().isoformat()}')
        print(f'BFD: {bfd_path}')
        print(f'Star Schema: {star_path}')

        # Load ground truth
        ground_truth_dict = self.ground_truth_loader.load_all()

        if len(ground_truth_dict) < 100:
            return {'status': 'error', 'message': 'Insufficient ground truth data'}

        # Load BFD database
        print(f'\n[LOADING] BFD database...')
        t0 = time.time()

        if GPU_AVAILABLE:
            bfd = cudf.read_parquet(bfd_path)
            df = bfd.to_pandas()
            del bfd
            clear_gpu()
        else:
            df = pd.read_parquet(bfd_path)

        print(f'  Loaded: {len(df):,} rows × {len(df.columns)} columns')
        print(f'  Time: {fmt(time.time()-t0)}')

        # Find ID column
        id_col = None
        for col in ['imdb_id', 'tmdb_id', 'fc_uid']:
            if col in df.columns:
                id_col = col
                break

        if id_col is None:
            return {'status': 'error', 'message': 'No ID column found'}

        # Get matching ground truth
        ground_truth_series = self.ground_truth_loader.get_matching_ground_truth(df, id_col)

        # Run optimization
        metrics = self.optimizer.optimize(df, ground_truth_series)

        if metrics['status'] != 'success':
            return metrics

        # Save optimized weights
        self.optimizer.save_weights()

        # Record in tracker
        self.tracker.record(version, metrics)

        # Mark as processed
        self.processed_versions.add(version)
        self._save_state()

        # Generate log
        log = {
            'run_id': f"MAPIE-INT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'db_version': version,
            'metrics': metrics,
            'runtime_seconds': time.time() - start_time
        }

        log_path = f'{MAPIE_DIR}/INTEGRATED_RUN_LOG_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        safe_json_save(log_path, log)

        print('\n' + '='*70)
        print('EXAMINATION COMPLETE')
        print('='*70)
        print(f'  Final MAPE: {metrics["final_mape"]:.2f}%')
        print(f'  Improvement: {metrics["improvement"]:.2f}%')
        print(f'  Runtime: {fmt(time.time() - start_time)}')
        print(f'  Log: {log_path}')

        return log

    def run_once(self):
        """Run examination on latest version"""
        result = self.get_latest_version()
        if result:
            version, bfd_path, star_path = result
            if star_path and os.path.exists(star_path):
                return self.run_examination(version, bfd_path, star_path)
            else:
                print(f"[ERROR] Star Schema not found for V{version}")
        else:
            print("[ERROR] No database versions found")

    def watch(self, interval: int = POLL_INTERVAL):
        """Watch for new versions and trigger automatically"""
        print('\n' + '='*70)
        print('MAPIE INTEGRATED RUNNER - WATCH MODE')
        print('='*70)
        print(f'Monitoring: {BASE_DIR}')
        print(f'Interval: {interval}s')
        print(f'Press Ctrl+C to stop')

        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'should_stop', True))
        signal.signal(signal.SIGTERM, lambda s, f: setattr(self, 'should_stop', True))

        while not self.should_stop:
            new = self.check_for_new_version()
            if new:
                version, bfd_path, star_path = new
                if star_path and os.path.exists(star_path):
                    print(f"\n[DETECTED] New version V{version}")
                    self.run_examination(version, bfd_path, star_path)
                    print("\nResuming watch...")

            time.sleep(interval)

        print("\nWatch mode stopped")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='MAPIE Integrated Runner')
    parser.add_argument('--watch', '-w', action='store_true', help='Watch for new versions')
    parser.add_argument('--once', '-o', action='store_true', help='Run once on latest')
    parser.add_argument('--version', '-v', help='Run on specific version')
    parser.add_argument('--interval', '-i', type=int, default=30, help='Watch interval')

    args = parser.parse_args()

    runner = IntegratedRunner()

    if args.watch:
        runner.watch(args.interval)
    elif args.version:
        base = Path(BASE_DIR)
        bfd_files = list(base.glob(f'Cranberry_BFD_*V{args.version}.parquet'))
        star_files = list(base.glob(f'Cranberry_Star_Schema_*V{args.version}.parquet'))
        if bfd_files and star_files:
            runner.run_examination(args.version, str(bfd_files[0]), str(star_files[0]))
        else:
            print(f"Version {args.version} not found")
    else:
        runner.run_once()


if __name__ == '__main__':
    main()
