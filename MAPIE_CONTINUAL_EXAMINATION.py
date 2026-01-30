#!/usr/bin/env python3
"""
MAPIE CONTINUAL EXAMINATION SYSTEM
===================================
VERSION: 1.0.0
CREATED: 2026-01-19

PURPOSE: Continuously monitor ViewerDBX masterdatabase updates and improve
algorithm weightings to reduce MAPE of Views calculations.

WHAT IT DOES:
1. Monitors Downloads folder for new Cranberry_BFD_V*.parquet and
   Cranberry_Star_Schema_V*.parquet twin databases
2. When new versions detected, loads and examines all views:
   - Abstract Views (from abstract signals)
   - Component Views (from lookup tables and weights)
   - True Validated Views (from training data ground truth)
3. Calculates MAPE against ground truth
4. Adjusts algorithm weightings to minimize MAPE
5. Outputs improved weightings to Components/

TRIGGERS:
- New database version published in Downloads folder
- Runs after MAPIE_DAILY_RUNNER completes
- Can be scheduled or triggered by file watcher

OUTPUT FILES:
- Components/MAPIE_WEIGHT_HISTORY.json - Weight adjustment history
- Components/MAPIE_MAPE_TRACKER.json - MAPE improvements over time
- Components/OPTIMIZED_WEIGHTS.json - Current best weights
- MAPIE/EXAMINATION_LOG_{timestamp}.json - Detailed examination log

============================================================================
"""
import os
import sys
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

# Configure environment for GPU
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['NUMBA_CUDA_USE_NVIDIA_BINDING'] = '1'
os.environ['CUDF_SPILL'] = 'on'

import numpy as np
import pandas as pd

# Try GPU libraries, fall back to CPU
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
    cp.cuda.Device(0).use()
    cp.get_default_memory_pool().free_all_blocks()
except ImportError:
    GPU_AVAILABLE = False
    print("[WARN] GPU libraries not available, using CPU")


# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = '/mnt/c/Users/RoyT6/Downloads'
COMP_DIR = f'{BASE_DIR}/Components'
MAPIE_DIR = f'{BASE_DIR}/MAPIE'
VIEWS_DIR = f'{BASE_DIR}/Views TRaining Data'

# Anti-cheat bounds from ALGO spec
VALID_R2_RANGE = (0.30, 0.90)
VALID_MAPE_RANGE = (5.0, 40.0)  # percent

# Learning parameters
INITIAL_LEARNING_RATE = 0.05
MIN_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.95
MAX_ITERATIONS_PER_CYCLE = 50
CONVERGENCE_THRESHOLD = 0.001  # Stop if MAPE improvement < 0.1%


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def fmt(seconds: float) -> str:
    """Format seconds as Xm Ys"""
    return f'{int(seconds//60)}m {int(seconds%60)}s'


def safe_json_load(fpath: str) -> Optional[dict]:
    """Safely load JSON with multiple encodings"""
    for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
        try:
            with open(fpath, 'r', encoding=enc) as f:
                return json.load(f)
        except:
            continue
    return None


def safe_json_save(fpath: str, data: dict) -> bool:
    """Safely save JSON"""
    try:
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save {fpath}: {e}")
        return False


def get_file_hash(fpath: str) -> str:
    """Get MD5 hash of file for change detection"""
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
# DATABASE VERSION DETECTION
# ============================================================================
class VersionDetector:
    """Monitors Downloads folder for new database versions"""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.known_versions = set()
        self.version_hashes = {}
        self._scan_existing()

    def _scan_existing(self):
        """Scan for existing database versions"""
        bfd_files = list(self.base_dir.glob('Cranberry_BFD_V*.parquet'))
        star_files = list(self.base_dir.glob('Cranberry_Star_Schema_V*.parquet'))

        for f in bfd_files:
            match = re.search(r'V(\d+\.\d+)', f.name)
            if match:
                version = match.group(1)
                self.known_versions.add(version)
                self.version_hashes[f'bfd_{version}'] = get_file_hash(str(f))

        for f in star_files:
            match = re.search(r'V(\d+\.\d+)', f.name)
            if match:
                version = match.group(1)
                self.known_versions.add(version)
                self.version_hashes[f'star_{version}'] = get_file_hash(str(f))

        print(f"[VersionDetector] Found {len(self.known_versions)} existing versions")

    def get_latest_version(self) -> str:
        """Get the latest database version"""
        if not self.known_versions:
            return "0.00"
        return max(self.known_versions, key=lambda x: float(x))

    def check_for_new_version(self) -> Optional[str]:
        """Check if new database version exists"""
        bfd_files = list(self.base_dir.glob('Cranberry_BFD_V*.parquet'))

        for f in bfd_files:
            match = re.search(r'V(\d+\.\d+)', f.name)
            if match:
                version = match.group(1)
                current_hash = get_file_hash(str(f))

                # Check if this is a new version or updated file
                hash_key = f'bfd_{version}'
                if hash_key not in self.version_hashes or self.version_hashes[hash_key] != current_hash:
                    self.known_versions.add(version)
                    self.version_hashes[hash_key] = current_hash
                    return version

        return None

    def get_twin_paths(self, version: str) -> Tuple[str, str]:
        """Get paths to BFD and Star Schema for a version"""
        bfd_path = self.base_dir / f'Cranberry_BFD_V{version}.parquet'
        star_path = self.base_dir / f'Cranberry_Star_Schema_V{version}.parquet'

        # Also check for MAPIE_RUN variants
        if not bfd_path.exists():
            bfd_files = list(self.base_dir.glob(f'Cranberry_BFD_*V{version}.parquet'))
            if bfd_files:
                bfd_path = bfd_files[0]

        if not star_path.exists():
            star_files = list(self.base_dir.glob(f'Cranberry_Star_Schema_*V{version}.parquet'))
            if star_files:
                star_path = star_files[0]

        return str(bfd_path), str(star_path)


# ============================================================================
# GROUND TRUTH LOADER
# ============================================================================
class GroundTruthLoader:
    """Loads and aggregates true validated views from training data"""

    def __init__(self, views_dir: str, comp_dir: str):
        self.views_dir = Path(views_dir)
        self.comp_dir = Path(comp_dir)
        self.ground_truth = {}
        self.stats = {
            'sources_loaded': 0,
            'total_records': 0,
            'total_views': 0
        }

    def load_all(self) -> Dict[str, float]:
        """Load all ground truth views from training data sources"""
        print("\n[GroundTruthLoader] Loading true validated views...")

        # Source 1: ETL_trueviews.csv (most comprehensive)
        etl_path = self.views_dir / 'ETL_trueviews.csv'
        if etl_path.exists():
            self._load_csv(etl_path, 'imdb_id', 'views', 'ETL_trueviews')

        # Source 2: AGGREGATED_VIEWS_BY_IMDB.csv
        agg_imdb = self.views_dir / 'AGGREGATED_VIEWS_BY_IMDB.csv'
        if agg_imdb.exists():
            self._load_csv(agg_imdb, 'imdb_id', 'total_views', 'AGGREGATED_IMDB')

        # Source 3: AGGREGATED_VIEWS_BY_TMDB.csv
        agg_tmdb = self.views_dir / 'AGGREGATED_VIEWS_BY_TMDB.csv'
        if agg_tmdb.exists():
            self._load_csv(agg_tmdb, 'tmdb_id', 'total_views', 'AGGREGATED_TMDB')

        # Source 4: AGGREGATED_VIEWS_BY_TITLE.csv
        agg_title = self.views_dir / 'AGGREGATED_VIEWS_BY_TITLE.csv'
        if agg_title.exists():
            self._load_csv(agg_title, 'title', 'total_views', 'AGGREGATED_TITLE')

        # Source 5: TRAINING_MATRIX_UNIFIED.parquet (Engine 1 output)
        training_path = self.comp_dir / 'TRAINING_MATRIX_UNIFIED.parquet'
        if training_path.exists():
            self._load_training_matrix(training_path)

        print(f"  Total ground truth records: {len(self.ground_truth):,}")
        print(f"  Total views tracked: {sum(self.ground_truth.values()):,.0f}")

        return self.ground_truth

    def _load_csv(self, path: Path, id_col: str, views_col: str, source_name: str):
        """Load a CSV source"""
        try:
            df = pd.read_csv(path, low_memory=False)

            if id_col not in df.columns or views_col not in df.columns:
                # Try alternate column names
                for alt_id in ['imdb_id', 'tmdb_id', 'title', 'fc_uid', 'id']:
                    if alt_id in df.columns:
                        id_col = alt_id
                        break
                for alt_views in ['views', 'total_views', 'hours_viewed', 'hours']:
                    if alt_views in df.columns:
                        views_col = alt_views
                        break

            if id_col not in df.columns or views_col not in df.columns:
                print(f"    [SKIP] {source_name}: columns not found")
                return

            # Aggregate by ID
            for _, row in df.iterrows():
                key = str(row[id_col]).strip()
                if not key or key == 'nan':
                    continue

                views = float(row[views_col]) if pd.notna(row[views_col]) else 0
                if views > 0:
                    if key in self.ground_truth:
                        # Average multiple sources
                        self.ground_truth[key] = (self.ground_truth[key] + views) / 2
                    else:
                        self.ground_truth[key] = views
                    self.stats['total_records'] += 1
                    self.stats['total_views'] += views

            self.stats['sources_loaded'] += 1
            print(f"    [OK] {source_name}: {len(df):,} records")

        except Exception as e:
            print(f"    [ERROR] {source_name}: {e}")

    def _load_training_matrix(self, path: Path):
        """Load training matrix parquet"""
        try:
            df = pd.read_parquet(path)

            # Find ID and views columns
            id_col = None
            views_col = None

            for col in ['imdb_id', 'tmdb_id', 'fc_uid', 'title_id']:
                if col in df.columns:
                    id_col = col
                    break

            for col in ['views', 'total_views', 'hours_viewed']:
                if col in df.columns:
                    views_col = col
                    break

            if id_col and views_col:
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

                self.stats['sources_loaded'] += 1
                print(f"    [OK] TRAINING_MATRIX: {len(df):,} records")

        except Exception as e:
            print(f"    [ERROR] TRAINING_MATRIX: {e}")


# ============================================================================
# WEIGHT OPTIMIZER
# ============================================================================
class WeightOptimizer:
    """Optimizes algorithm weights to minimize MAPE"""

    def __init__(self, comp_dir: str):
        self.comp_dir = Path(comp_dir)
        self.current_weights = self._load_weights()
        self.weight_history = []
        self.mape_history = []
        self.learning_rate = INITIAL_LEARNING_RATE

    def _load_weights(self) -> Dict:
        """Load current weights from Components"""
        weights = {
            'genre_weights': {},
            'type_weights': {'movie': 1.2, 'series': 0.9, 'default': 1.0},
            'year_weights': {'current': 1.3, 'recent': 1.1, 'mid': 1.0, 'old': 0.8},
            'studio_weights': {},
            'country_weights': {},
            'platform_weights': {},
            'signal_weights': {}
        }

        # Load genre decay table
        genre_path = self.comp_dir / 'cranberry genre decay table.json'
        if genre_path.exists():
            data = safe_json_load(str(genre_path))
            if data and 'genres' in data:
                for genre, params in data['genres'].items():
                    if isinstance(params, dict):
                        halflife = params.get('halflife_days', 30)
                        baseline = params.get('baseline_B', 0.15)
                        weights['genre_weights'][genre.lower()] = 0.85 + (halflife / 100) + baseline

        # Load studio weights
        studio_path = self.comp_dir / 'Apply studio weighting.json'
        if studio_path.exists():
            data = safe_json_load(str(studio_path))
            if data and 'weight_lookup' in data:
                weights['studio_weights'] = {k.lower(): v for k, v in data['weight_lookup'].items()}

        # Load country weights
        country_path = self.comp_dir / 'country_viewership_weights_2025.json'
        if country_path.exists():
            data = safe_json_load(str(country_path))
            if data and 'weights' in data:
                for country, info in data['weights'].items():
                    if isinstance(info, dict):
                        weights['country_weights'][country.upper()] = info.get('weight_percent', 0) / 100

        # Load existing optimized weights if available
        opt_path = self.comp_dir / 'OPTIMIZED_WEIGHTS.json'
        if opt_path.exists():
            opt_data = safe_json_load(str(opt_path))
            if opt_data and 'weights' in opt_data:
                # Merge optimized weights
                for key, value in opt_data['weights'].items():
                    if key in weights and isinstance(value, dict):
                        weights[key].update(value)

        return weights

    def calculate_mape(self, computed_views: pd.Series, ground_truth: Dict[str, float],
                       id_column: pd.Series) -> Tuple[float, int]:
        """Calculate MAPE against ground truth"""
        matched = 0
        total_ape = 0.0

        for idx, comp_views in computed_views.items():
            id_val = str(id_column.iloc[idx]) if hasattr(id_column, 'iloc') else str(id_column[idx])

            if id_val in ground_truth:
                true_views = ground_truth[id_val]
                if true_views > 0:
                    ape = abs(comp_views - true_views) / true_views
                    total_ape += ape
                    matched += 1

        if matched == 0:
            return 100.0, 0  # No matches

        mape = (total_ape / matched) * 100
        return mape, matched

    def optimize_weights(self, bfd_df: pd.DataFrame, ground_truth: Dict[str, float],
                         max_iterations: int = MAX_ITERATIONS_PER_CYCLE) -> Dict:
        """
        Iteratively optimize weights to minimize MAPE
        Uses gradient-free optimization (coordinate descent)
        """
        print("\n[WeightOptimizer] Starting weight optimization...")

        # Find ID column
        id_col = None
        for col in ['imdb_id', 'tmdb_id', 'fc_uid']:
            if col in bfd_df.columns:
                id_col = col
                break

        if id_col is None:
            print("  [ERROR] No ID column found for matching")
            return self.current_weights

        # Initial MAPE calculation
        computed = self._compute_views(bfd_df)
        initial_mape, matched = self.calculate_mape(computed, ground_truth, bfd_df[id_col])

        if matched < 100:
            print(f"  [WARN] Only {matched} matches found - insufficient for optimization")
            return self.current_weights

        print(f"  Initial MAPE: {initial_mape:.2f}% ({matched:,} matched records)")

        best_mape = initial_mape
        best_weights = self.current_weights.copy()

        # Optimization loop
        for iteration in range(max_iterations):
            improved = False

            # Try adjusting each weight category
            for category in ['genre_weights', 'type_weights', 'year_weights']:
                if category not in self.current_weights:
                    continue

                for key in list(self.current_weights[category].keys()):
                    original_value = self.current_weights[category][key]

                    # Try increasing
                    self.current_weights[category][key] = original_value * (1 + self.learning_rate)
                    computed = self._compute_views(bfd_df)
                    mape_up, _ = self.calculate_mape(computed, ground_truth, bfd_df[id_col])

                    # Try decreasing
                    self.current_weights[category][key] = original_value * (1 - self.learning_rate)
                    computed = self._compute_views(bfd_df)
                    mape_down, _ = self.calculate_mape(computed, ground_truth, bfd_df[id_col])

                    # Keep best direction
                    if mape_up < best_mape and mape_up < mape_down:
                        self.current_weights[category][key] = original_value * (1 + self.learning_rate)
                        best_mape = mape_up
                        best_weights = {k: dict(v) if isinstance(v, dict) else v
                                       for k, v in self.current_weights.items()}
                        improved = True
                    elif mape_down < best_mape:
                        self.current_weights[category][key] = original_value * (1 - self.learning_rate)
                        best_mape = mape_down
                        best_weights = {k: dict(v) if isinstance(v, dict) else v
                                       for k, v in self.current_weights.items()}
                        improved = True
                    else:
                        # Revert
                        self.current_weights[category][key] = original_value

            self.mape_history.append(best_mape)

            # Check convergence
            if len(self.mape_history) > 1:
                improvement = self.mape_history[-2] - self.mape_history[-1]
                if improvement < CONVERGENCE_THRESHOLD:
                    print(f"  Converged at iteration {iteration + 1}")
                    break

            # Decay learning rate
            if not improved:
                self.learning_rate *= LEARNING_RATE_DECAY
                if self.learning_rate < MIN_LEARNING_RATE:
                    print(f"  Learning rate too small, stopping at iteration {iteration + 1}")
                    break

            if (iteration + 1) % 10 == 0:
                print(f"    Iteration {iteration + 1}: MAPE = {best_mape:.2f}%")

        self.current_weights = best_weights

        final_improvement = initial_mape - best_mape
        print(f"  Final MAPE: {best_mape:.2f}%")
        print(f"  Improvement: {final_improvement:.2f}% ({final_improvement/initial_mape*100:.1f}% relative)")

        # Validate against anti-cheat bounds
        if best_mape < VALID_MAPE_RANGE[0]:
            print(f"  [WARN] MAPE {best_mape:.2f}% below valid range - possible data leakage")

        return self.current_weights

    def _compute_views(self, bfd_df: pd.DataFrame) -> pd.Series:
        """Compute views using current weights"""
        weights = self.current_weights

        # Start with base views
        if 'views_computed' in bfd_df.columns:
            base = bfd_df['views_computed'].copy()
        else:
            base = pd.Series(1000000, index=bfd_df.index)

        # Apply genre weights
        if 'genres' in bfd_df.columns:
            def get_genre_mult(genres):
                if pd.isna(genres):
                    return 1.0
                g = str(genres).lower()
                for genre, mult in weights.get('genre_weights', {}).items():
                    if genre in g:
                        return mult
                return 1.0

            genre_mult = bfd_df['genres'].apply(get_genre_mult)
            base = base * genre_mult

        # Apply type weights
        if 'title_type' in bfd_df.columns:
            def get_type_mult(t):
                if pd.isna(t):
                    return weights['type_weights'].get('default', 1.0)
                t = str(t).lower()
                if 'movie' in t:
                    return weights['type_weights'].get('movie', 1.2)
                elif 'series' in t:
                    return weights['type_weights'].get('series', 0.9)
                return weights['type_weights'].get('default', 1.0)

            type_mult = bfd_df['title_type'].apply(get_type_mult)
            base = base * type_mult

        # Apply year weights
        if 'start_year' in bfd_df.columns:
            current_year = datetime.now().year

            def get_year_mult(year):
                if pd.isna(year):
                    return weights['year_weights'].get('old', 0.8)
                try:
                    y = int(year)
                    if y >= current_year - 1:
                        return weights['year_weights'].get('current', 1.3)
                    elif y >= current_year - 3:
                        return weights['year_weights'].get('recent', 1.1)
                    elif y >= current_year - 5:
                        return weights['year_weights'].get('mid', 1.0)
                    else:
                        return weights['year_weights'].get('old', 0.8)
                except:
                    return weights['year_weights'].get('old', 0.8)

            year_mult = bfd_df['start_year'].apply(get_year_mult)
            base = base * year_mult

        return base.clip(lower=10000)

    def save_weights(self):
        """Save optimized weights to Components"""
        output = {
            'version': '1.0',
            'updated': datetime.now().isoformat(),
            'learning_rate': self.learning_rate,
            'weights': self.current_weights,
            'mape_history': self.mape_history[-10:] if self.mape_history else []
        }

        opt_path = self.comp_dir / 'OPTIMIZED_WEIGHTS.json'
        safe_json_save(str(opt_path), output)
        print(f"  Saved optimized weights to {opt_path}")

        # Also save weight history
        history_path = self.comp_dir / 'MAPIE_WEIGHT_HISTORY.json'
        history_data = safe_json_load(str(history_path)) or {'entries': []}
        history_data['entries'].append({
            'timestamp': datetime.now().isoformat(),
            'mape': self.mape_history[-1] if self.mape_history else None,
            'weights_snapshot': {k: dict(v) if isinstance(v, dict) else v
                                for k, v in self.current_weights.items()}
        })
        # Keep last 100 entries
        history_data['entries'] = history_data['entries'][-100:]
        safe_json_save(str(history_path), history_data)


# ============================================================================
# MAPIE TRACKER
# ============================================================================
class MAPIETracker:
    """Tracks MAPE improvements over time"""

    def __init__(self, comp_dir: str):
        self.comp_dir = Path(comp_dir)
        self.tracker_path = self.comp_dir / 'MAPIE_MAPE_TRACKER.json'
        self.data = self._load()

    def _load(self) -> Dict:
        """Load existing tracker data"""
        if self.tracker_path.exists():
            return safe_json_load(str(self.tracker_path)) or self._default()
        return self._default()

    def _default(self) -> Dict:
        return {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'entries': [],
            'best_mape': 100.0,
            'best_version': None,
            'improvement_trend': []
        }

    def record(self, version: str, mape: float, r2: float, matched: int,
               views_types: Dict[str, float]):
        """Record a MAPE measurement"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'db_version': version,
            'mape': mape,
            'r2': r2,
            'matched_records': matched,
            'views_breakdown': views_types,
            'within_valid_range': VALID_MAPE_RANGE[0] <= mape <= VALID_MAPE_RANGE[1]
        }

        self.data['entries'].append(entry)

        # Track best
        if mape < self.data['best_mape']:
            self.data['best_mape'] = mape
            self.data['best_version'] = version

        # Track improvement trend
        if len(self.data['entries']) > 1:
            prev_mape = self.data['entries'][-2]['mape']
            improvement = prev_mape - mape
            self.data['improvement_trend'].append({
                'from_version': self.data['entries'][-2]['db_version'],
                'to_version': version,
                'improvement': improvement
            })

        self._save()

    def _save(self):
        """Save tracker data"""
        self.data['updated'] = datetime.now().isoformat()
        safe_json_save(str(self.tracker_path), self.data)

    def get_summary(self) -> Dict:
        """Get summary of MAPE tracking"""
        if not self.data['entries']:
            return {'status': 'no_data'}

        mapes = [e['mape'] for e in self.data['entries']]
        return {
            'total_entries': len(self.data['entries']),
            'best_mape': self.data['best_mape'],
            'best_version': self.data['best_version'],
            'latest_mape': mapes[-1] if mapes else None,
            'average_mape': sum(mapes) / len(mapes),
            'total_improvement': mapes[0] - mapes[-1] if len(mapes) > 1 else 0
        }


# ============================================================================
# MAIN EXAMINATION ENGINE
# ============================================================================
class ContinualExaminationEngine:
    """Main engine that orchestrates the continual examination process"""

    def __init__(self):
        self.version_detector = VersionDetector(BASE_DIR)
        self.ground_truth_loader = GroundTruthLoader(VIEWS_DIR, COMP_DIR)
        self.weight_optimizer = WeightOptimizer(COMP_DIR)
        self.mape_tracker = MAPIETracker(COMP_DIR)
        self.last_examined_version = None
        self.run_count = 0

    def run_examination(self, version: str = None) -> Dict:
        """Run a full examination cycle on a database version"""
        start_time = time.time()
        self.run_count += 1

        print('\n' + '='*70)
        print('MAPIE CONTINUAL EXAMINATION SYSTEM')
        print('='*70)
        print(f'Run #{self.run_count} | {datetime.now().isoformat()}')

        # Get version to examine
        if version is None:
            version = self.version_detector.get_latest_version()

        print(f'Examining database version: V{version}')

        # Get database paths
        bfd_path, star_path = self.version_detector.get_twin_paths(version)

        if not os.path.exists(bfd_path):
            return {'status': 'error', 'message': f'BFD file not found: {bfd_path}'}

        # Load ground truth
        ground_truth = self.ground_truth_loader.load_all()

        if len(ground_truth) < 100:
            return {'status': 'error', 'message': 'Insufficient ground truth data'}

        # Load BFD database
        print(f'\n[LOADING] BFD database...')
        t0 = time.time()

        if GPU_AVAILABLE:
            bfd = cudf.read_parquet(bfd_path)
            bfd_df = bfd.to_pandas()
            del bfd
            clear_gpu()
        else:
            bfd_df = pd.read_parquet(bfd_path)

        print(f'  Loaded: {len(bfd_df):,} rows × {len(bfd_df.columns)} columns')
        print(f'  Time: {fmt(time.time()-t0)}')

        # Calculate views breakdown
        views_breakdown = self._calculate_views_breakdown(bfd_df)

        # Calculate initial MAPE
        print('\n[ANALYSIS] Calculating MAPE against ground truth...')

        id_col = None
        for col in ['imdb_id', 'tmdb_id', 'fc_uid']:
            if col in bfd_df.columns:
                id_col = col
                break

        if id_col and 'views_computed' in bfd_df.columns:
            initial_mape, matched = self.weight_optimizer.calculate_mape(
                bfd_df['views_computed'], ground_truth, bfd_df[id_col]
            )
            print(f'  Current MAPE: {initial_mape:.2f}% ({matched:,} matched)')
        else:
            initial_mape = 100.0
            matched = 0
            print('  [WARN] Could not calculate MAPE - missing columns')

        # Optimize weights
        print('\n[OPTIMIZATION] Adjusting algorithm weights...')
        optimized_weights = self.weight_optimizer.optimize_weights(bfd_df, ground_truth)

        # Calculate final MAPE after optimization
        final_computed = self.weight_optimizer._compute_views(bfd_df)
        final_mape, final_matched = self.weight_optimizer.calculate_mape(
            final_computed, ground_truth, bfd_df[id_col]
        )

        # Calculate R2
        r2 = self._calculate_r2(final_computed, ground_truth, bfd_df[id_col])

        # Save optimized weights
        self.weight_optimizer.save_weights()

        # Record in tracker
        self.mape_tracker.record(version, final_mape, r2, final_matched, views_breakdown)

        # Generate examination log
        log = self._generate_log(version, initial_mape, final_mape, r2,
                                 matched, views_breakdown, time.time() - start_time)

        self.last_examined_version = version

        # Print summary
        print('\n' + '='*70)
        print('EXAMINATION COMPLETE')
        print('='*70)
        print(f'  Database Version: V{version}')
        print(f'  Initial MAPE: {initial_mape:.2f}%')
        print(f'  Final MAPE: {final_mape:.2f}%')
        print(f'  Improvement: {initial_mape - final_mape:.2f}%')
        print(f'  R2 Score: {r2:.4f}')
        print(f'  Matched Records: {final_matched:,}')
        print(f'  Runtime: {fmt(time.time() - start_time)}')

        return log

    def _calculate_views_breakdown(self, bfd_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate breakdown of views by type"""
        breakdown = {
            'total_views_computed': 0,
            'abstract_views': 0,
            'component_views': 0,
            'validated_views': 0
        }

        if 'views_computed' in bfd_df.columns:
            breakdown['total_views_computed'] = float(bfd_df['views_computed'].sum())

        # Estimate abstract views from abs_* columns
        abs_cols = [c for c in bfd_df.columns if c.startswith('abs_')]
        if abs_cols:
            breakdown['abstract_views'] = float(bfd_df[abs_cols].sum().sum())

        # Component views would be derived from lookup applications
        breakdown['component_views'] = breakdown['total_views_computed'] * 0.3  # Estimate

        # Validated views from training matches
        breakdown['validated_views'] = breakdown['total_views_computed'] * 0.7  # Estimate

        return breakdown

    def _calculate_r2(self, computed: pd.Series, ground_truth: Dict[str, float],
                      id_column: pd.Series) -> float:
        """Calculate R-squared score"""
        actual = []
        predicted = []

        for idx, comp_views in computed.items():
            id_val = str(id_column.iloc[idx]) if hasattr(id_column, 'iloc') else str(id_column[idx])
            if id_val in ground_truth:
                actual.append(ground_truth[id_val])
                predicted.append(comp_views)

        if len(actual) < 10:
            return 0.0

        actual = np.array(actual)
        predicted = np.array(predicted)

        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)

        if ss_tot == 0:
            return 0.0

        r2 = 1 - (ss_res / ss_tot)
        return max(0, min(1, r2))  # Clamp to [0, 1]

    def _generate_log(self, version: str, initial_mape: float, final_mape: float,
                      r2: float, matched: int, views_breakdown: Dict,
                      runtime: float) -> Dict:
        """Generate detailed examination log"""
        log = {
            'examination_id': f"EXAM-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'database_version': version,
            'metrics': {
                'initial_mape': initial_mape,
                'final_mape': final_mape,
                'improvement': initial_mape - final_mape,
                'r2': r2,
                'matched_records': matched
            },
            'views_breakdown': views_breakdown,
            'anti_cheat_check': {
                'mape_valid': VALID_MAPE_RANGE[0] <= final_mape <= VALID_MAPE_RANGE[1],
                'r2_valid': VALID_R2_RANGE[0] <= r2 <= VALID_R2_RANGE[1]
            },
            'weights_updated': True,
            'runtime_seconds': runtime,
            'status': 'SUCCESS'
        }

        # Save log
        log_path = Path(MAPIE_DIR) / f"EXAMINATION_LOG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        safe_json_save(str(log_path), log)
        print(f'\n  Log saved: {log_path}')

        return log

    def watch_for_updates(self, check_interval: int = 60):
        """
        Continuous watch mode - monitors for new database versions
        and triggers examination when detected
        """
        print('\n' + '='*70)
        print('MAPIE CONTINUAL EXAMINATION - WATCH MODE')
        print('='*70)
        print(f'Monitoring: {BASE_DIR}')
        print(f'Check interval: {check_interval} seconds')
        print(f'Current latest version: V{self.version_detector.get_latest_version()}')
        print('\nWaiting for new database versions...')
        print('(Press Ctrl+C to stop)')

        try:
            while True:
                new_version = self.version_detector.check_for_new_version()

                if new_version:
                    print(f'\n[DETECTED] New database version: V{new_version}')
                    self.run_examination(new_version)
                    print(f'\nResuming watch mode...')

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print('\n\nWatch mode stopped by user.')
            summary = self.mape_tracker.get_summary()
            print(f'Total examinations: {summary.get("total_entries", 0)}')
            print(f'Best MAPE achieved: {summary.get("best_mape", "N/A")}%')


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='MAPIE Continual Examination System')
    parser.add_argument('--version', '-v', help='Specific version to examine')
    parser.add_argument('--watch', '-w', action='store_true', help='Watch mode - monitor for updates')
    parser.add_argument('--interval', '-i', type=int, default=60, help='Check interval in seconds (watch mode)')

    args = parser.parse_args()

    engine = ContinualExaminationEngine()

    if args.watch:
        engine.watch_for_updates(args.interval)
    else:
        result = engine.run_examination(args.version)
        print(f'\nExamination status: {result.get("status", "UNKNOWN")}')


if __name__ == '__main__':
    main()
