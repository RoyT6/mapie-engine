#!/usr/bin/env python3
"""
MAPIE V28 - SPLIT DATABASE ARCHITECTURE
========================================
VERSION: 28.00
CREATED: 2026-01-27

PURPOSE: Updated MAPIE system for the new split database architecture:
  - BFD_META_V27.72.parquet   (514K rows × 1407 cols) - Core metadata, predictions
  - BFD_ML_V27.72.parquet     (514K rows × 398 cols)  - ML features
  - BFD_VIEWS_V27.72.parquet  (514K rows × 493 cols)  - Ground truth views
  - CREATIVE_TALENT_V27.66.parquet (74K rows)         - Cast/crew
  - SEASON_AGGREGATES_V27.66.parquet (150K rows)      - Season data

CHANGES FROM V27:
  - Single Cranberry_BFD.parquet → Split into 5 specialized files
  - Join on fc_uid across META, ML, VIEWS tables
  - Ground truth now in BFD_VIEWS (netflix_views, views_estimated)
  - Abstract signals and predictions remain in BFD_META

============================================================================
"""
import os
import sys
import time
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure environment
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['NUMBA_CUDA_USE_NVIDIA_BINDING'] = '1'
os.environ['CUDF_SPILL'] = 'on'

import numpy as np
import pandas as pd

# Try GPU libraries
GPU_AVAILABLE = False
try:
    import cudf
    import cupy as cp
    cp.cuda.Device(0).use()
    GPU_AVAILABLE = True
    print("[INFO] GPU acceleration enabled")
except Exception as e:
    GPU_AVAILABLE = False
    print(f"[INFO] Using CPU mode (GPU not available: {type(e).__name__})")


# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = '/mnt/c/Users/RoyT6/Downloads'
COMP_DIR = f'{BASE_DIR}/Components'
MAPIE_DIR = f'{BASE_DIR}/MAPIE'

# New split database files
DB_FILES = {
    'meta': 'BFD_META_V27.72.parquet',
    'ml': 'BFD_ML_V27.72.parquet',
    'views': 'BFD_VIEWS_V27.72.parquet',
    'talent': 'CREATIVE_TALENT_V27.66.parquet',
    'seasons': 'SEASON_AGGREGATES_V27.66.parquet'
}

# Anti-cheat bounds
VALID_MAPE_RANGE = (5.0, 40.0)
VALID_R2_RANGE = (0.30, 0.90)

# Learning parameters
INITIAL_LEARNING_RATE = 0.05
MIN_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.95
MAX_ITERATIONS = 50
CONVERGENCE_THRESHOLD = 0.001


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def fmt(seconds: float) -> str:
    """Format seconds as Xm Ys"""
    return f'{int(seconds//60)}m {int(seconds%60)}s'


def safe_json_load(fpath: str) -> Optional[dict]:
    """Safely load JSON"""
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


def clear_gpu():
    """Clear GPU memory"""
    if GPU_AVAILABLE:
        import gc
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


# ============================================================================
# DATABASE LOADER - SPLIT ARCHITECTURE
# ============================================================================
class SplitDatabaseLoader:
    """Loads and joins the split BFD database files"""

    def __init__(self, base_dir: str = BASE_DIR):
        self.base_dir = Path(base_dir)
        self.meta_df = None
        self.ml_df = None
        self.views_df = None
        self.talent_df = None
        self.seasons_df = None
        self.stats = {}

    def detect_versions(self) -> Dict[str, str]:
        """Auto-detect latest versions of each database file"""
        versions = {}
        patterns = {
            'meta': 'BFD_META_V*.parquet',
            'ml': 'BFD_ML_V*.parquet',
            'views': 'BFD_VIEWS_V*.parquet',
            'talent': 'CREATIVE_TALENT_V*.parquet',
            'seasons': 'SEASON_AGGREGATES_V*.parquet'
        }

        for key, pattern in patterns.items():
            files = list(self.base_dir.glob(pattern))
            if files:
                # Get latest version
                latest = max(files, key=lambda f: self._extract_version(f.name))
                versions[key] = latest.name

        return versions

    def _extract_version(self, filename: str) -> float:
        """Extract version number from filename"""
        match = re.search(r'V(\d+\.\d+)', filename)
        return float(match.group(1)) if match else 0.0

    def load_all(self, files: Dict[str, str] = None) -> pd.DataFrame:
        """Load all database files and return joined DataFrame"""
        if files is None:
            files = self.detect_versions()

        print("\n" + "="*70)
        print("LOADING SPLIT DATABASE FILES")
        print("="*70)

        t_start = time.time()

        # Load META (required)
        meta_path = self.base_dir / files.get('meta', DB_FILES['meta'])
        print(f"\n[1/5] Loading BFD_META: {meta_path.name}")
        t0 = time.time()
        if GPU_AVAILABLE:
            self.meta_df = cudf.read_parquet(str(meta_path)).to_pandas()
            clear_gpu()
        else:
            self.meta_df = pd.read_parquet(meta_path)
        print(f"      {len(self.meta_df):,} rows × {len(self.meta_df.columns)} cols [{fmt(time.time()-t0)}]")

        # Load VIEWS (for ground truth)
        views_path = self.base_dir / files.get('views', DB_FILES['views'])
        print(f"\n[2/5] Loading BFD_VIEWS: {views_path.name}")
        t0 = time.time()
        if GPU_AVAILABLE:
            self.views_df = cudf.read_parquet(str(views_path)).to_pandas()
            clear_gpu()
        else:
            self.views_df = pd.read_parquet(views_path)
        print(f"      {len(self.views_df):,} rows × {len(self.views_df.columns)} cols [{fmt(time.time()-t0)}]")

        # Load ML features (optional for basic MAPE)
        ml_path = self.base_dir / files.get('ml', DB_FILES['ml'])
        if ml_path.exists():
            print(f"\n[3/5] Loading BFD_ML: {ml_path.name}")
            t0 = time.time()
            if GPU_AVAILABLE:
                self.ml_df = cudf.read_parquet(str(ml_path)).to_pandas()
                clear_gpu()
            else:
                self.ml_df = pd.read_parquet(ml_path)
            print(f"      {len(self.ml_df):,} rows × {len(self.ml_df.columns)} cols [{fmt(time.time()-t0)}]")
        else:
            print(f"\n[3/5] BFD_ML not found, skipping")

        # Load TALENT (optional)
        talent_path = self.base_dir / files.get('talent', DB_FILES['talent'])
        if talent_path.exists():
            print(f"\n[4/5] Loading CREATIVE_TALENT: {talent_path.name}")
            t0 = time.time()
            self.talent_df = pd.read_parquet(talent_path)
            print(f"      {len(self.talent_df):,} rows × {len(self.talent_df.columns)} cols [{fmt(time.time()-t0)}]")
        else:
            print(f"\n[4/5] CREATIVE_TALENT not found, skipping")

        # Load SEASONS (optional)
        seasons_path = self.base_dir / files.get('seasons', DB_FILES['seasons'])
        if seasons_path.exists():
            print(f"\n[5/5] Loading SEASON_AGGREGATES: {seasons_path.name}")
            t0 = time.time()
            self.seasons_df = pd.read_parquet(seasons_path)
            print(f"      {len(self.seasons_df):,} rows × {len(self.seasons_df.columns)} cols [{fmt(time.time()-t0)}]")
        else:
            print(f"\n[5/5] SEASON_AGGREGATES not found, skipping")

        print(f"\n[DONE] Total load time: {fmt(time.time()-t_start)}")

        # Store stats
        self.stats = {
            'meta_rows': len(self.meta_df),
            'meta_cols': len(self.meta_df.columns),
            'views_rows': len(self.views_df) if self.views_df is not None else 0,
            'ml_rows': len(self.ml_df) if self.ml_df is not None else 0,
            'load_time': time.time() - t_start
        }

        return self.meta_df

    def get_joined_df(self, include_ml: bool = False) -> pd.DataFrame:
        """Get META joined with VIEWS (and optionally ML)"""
        if self.meta_df is None:
            self.load_all()

        # Select key columns from VIEWS for ground truth
        views_cols = ['fc_uid', 'netflix_views', 'views_estimated']
        views_cols = [c for c in views_cols if c in self.views_df.columns]

        # Merge
        df = self.meta_df.merge(
            self.views_df[views_cols],
            on='fc_uid',
            how='left',
            suffixes=('', '_views')
        )

        if include_ml and self.ml_df is not None:
            # Only include a subset of ML features to avoid memory issues
            ml_sample_cols = ['fc_uid'] + [c for c in self.ml_df.columns if 'lag7' in c or 'roll7' in c][:50]
            df = df.merge(
                self.ml_df[ml_sample_cols],
                on='fc_uid',
                how='left',
                suffixes=('', '_ml')
            )

        return df

    def get_ground_truth(self) -> Dict[str, float]:
        """Extract ground truth views from VIEWS table"""
        if self.views_df is None:
            self.load_all()

        ground_truth = {}

        # Priority 1: netflix_views
        if 'netflix_views' in self.views_df.columns:
            netflix_mask = self.views_df['netflix_views'].notna() & (self.views_df['netflix_views'] > 0)
            for _, row in self.views_df[netflix_mask].iterrows():
                ground_truth[str(row['fc_uid'])] = float(row['netflix_views'])

        # Priority 2: views_estimated (fill gaps)
        if 'views_estimated' in self.views_df.columns:
            est_mask = self.views_df['views_estimated'].notna() & (self.views_df['views_estimated'] > 0)
            for _, row in self.views_df[est_mask].iterrows():
                key = str(row['fc_uid'])
                if key not in ground_truth:
                    ground_truth[key] = float(row['views_estimated'])

        # Priority 3: views_y from META
        if self.meta_df is not None and 'views_y' in self.meta_df.columns:
            y_mask = self.meta_df['views_y'].notna() & (self.meta_df['views_y'] > 0)
            for _, row in self.meta_df[y_mask].iterrows():
                key = str(row['fc_uid'])
                if key not in ground_truth:
                    ground_truth[key] = float(row['views_y'])

        return ground_truth


# ============================================================================
# MAPE CALCULATOR
# ============================================================================
class MAPECalculator:
    """Calculate MAPE metrics for the split database"""

    def __init__(self, db_loader: SplitDatabaseLoader):
        self.db = db_loader

    def calculate_mape(self, computed_col: str = 'views_computed',
                       ground_truth: Dict[str, float] = None) -> Dict:
        """Calculate MAPE against ground truth"""
        if ground_truth is None:
            ground_truth = self.db.get_ground_truth()

        meta_df = self.db.meta_df
        if meta_df is None or computed_col not in meta_df.columns:
            return {'status': 'error', 'message': f'Column {computed_col} not found'}

        matched = 0
        total_ape = 0.0
        errors = []

        for idx, row in meta_df.iterrows():
            fc_uid = str(row['fc_uid'])
            if fc_uid not in ground_truth:
                continue

            computed = row[computed_col]
            actual = ground_truth[fc_uid]

            if pd.isna(computed) or computed <= 0 or actual <= 0:
                continue

            ape = abs(computed - actual) / actual
            total_ape += ape
            matched += 1
            errors.append({
                'fc_uid': fc_uid,
                'computed': computed,
                'actual': actual,
                'ape': ape
            })

        if matched == 0:
            return {'status': 'error', 'message': 'No matches found'}

        mape = (total_ape / matched) * 100

        # Calculate R²
        errors_df = pd.DataFrame(errors)
        computed_arr = errors_df['computed'].values
        actual_arr = errors_df['actual'].values

        ss_res = np.sum((actual_arr - computed_arr) ** 2)
        ss_tot = np.sum((actual_arr - np.mean(actual_arr)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2 = max(0, min(1, r2))

        return {
            'status': 'success',
            'mape': mape,
            'r2': r2,
            'matched': matched,
            'total_ground_truth': len(ground_truth),
            'match_rate': matched / len(ground_truth) * 100,
            'within_valid_range': VALID_MAPE_RANGE[0] <= mape <= VALID_MAPE_RANGE[1],
            'top_errors': sorted(errors, key=lambda x: x['ape'], reverse=True)[:10]
        }

    def compare_predictions(self) -> Dict:
        """Compare all prediction columns against ground truth"""
        meta_df = self.db.meta_df
        ground_truth = self.db.get_ground_truth()

        pred_cols = [
            'views_computed',
            'views_y_pred',
            'views_pred_xgboost',
            'views_pred_catboost',
            'views_pred_lightgbm',
            'views_pred_randomforest'
        ]

        results = {}
        for col in pred_cols:
            if col in meta_df.columns:
                result = self.calculate_mape(col, ground_truth)
                if result['status'] == 'success':
                    results[col] = {
                        'mape': result['mape'],
                        'r2': result['r2'],
                        'matched': result['matched']
                    }

        # Rank by MAPE
        ranked = sorted(results.items(), key=lambda x: x[1]['mape'])

        return {
            'results': results,
            'best_model': ranked[0][0] if ranked else None,
            'best_mape': ranked[0][1]['mape'] if ranked else None
        }


# ============================================================================
# WEIGHT OPTIMIZER
# ============================================================================
class WeightOptimizer:
    """Optimize algorithm weights to minimize MAPE"""

    def __init__(self, db_loader: SplitDatabaseLoader, comp_dir: str = COMP_DIR):
        self.db = db_loader
        self.comp_dir = Path(comp_dir)
        self.current_weights = self._load_weights()
        self.mape_history = []
        self.learning_rate = INITIAL_LEARNING_RATE

    def _load_weights(self) -> Dict:
        """Load current weights from Components"""
        weights = {
            'genre_weights': {},
            'type_weights': {'movie': 1.2, 'series': 0.9, 'default': 1.0},
            'year_weights': {'current': 1.3, 'recent': 1.1, 'mid': 1.0, 'old': 0.8},
            'studio_weights': {},
            'abstract_signal_weights': {}
        }

        # Load genre decay
        genre_path = self.comp_dir / 'cranberry genre decay table.json'
        if genre_path.exists():
            data = safe_json_load(str(genre_path))
            if data and 'genres' in data:
                for genre, params in data['genres'].items():
                    if isinstance(params, dict):
                        halflife = params.get('halflife_days', 30)
                        weights['genre_weights'][genre.lower()] = 0.85 + (halflife / 100)

        # Load existing optimized weights
        opt_path = self.comp_dir / 'OPTIMIZED_WEIGHTS.json'
        if opt_path.exists():
            opt_data = safe_json_load(str(opt_path))
            if opt_data and 'weights' in opt_data:
                for key, value in opt_data['weights'].items():
                    if key in weights and isinstance(value, dict):
                        weights[key].update(value)

        return weights

    def compute_views(self, df: pd.DataFrame) -> pd.Series:
        """Compute views using current weights"""
        weights = self.current_weights

        # Start with base prediction
        if 'base_prediction' in df.columns:
            base = df['base_prediction'].fillna(1000000)
        elif 'views_y_pred' in df.columns:
            base = df['views_y_pred'].fillna(1000000)
        else:
            base = pd.Series(1000000, index=df.index)

        # Apply genre weights
        if 'genres' in df.columns and weights.get('genre_weights'):
            def get_genre_mult(genres):
                if pd.isna(genres):
                    return 1.0
                g = str(genres).lower()
                for genre, mult in weights['genre_weights'].items():
                    if genre in g:
                        return mult
                return 1.0
            genre_mult = df['genres'].apply(get_genre_mult)
            base = base * genre_mult

        # Apply type weights
        if 'title_type' in df.columns:
            def get_type_mult(t):
                if pd.isna(t):
                    return weights['type_weights'].get('default', 1.0)
                t = str(t).lower()
                if 'movie' in t:
                    return weights['type_weights'].get('movie', 1.2)
                elif 'series' in t:
                    return weights['type_weights'].get('series', 0.9)
                return weights['type_weights'].get('default', 1.0)
            type_mult = df['title_type'].apply(get_type_mult)
            base = base * type_mult

        # Apply year weights
        if 'start_year' in df.columns:
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
            year_mult = df['start_year'].apply(get_year_mult)
            base = base * year_mult

        return base.clip(lower=10000)

    def optimize(self, max_iterations: int = MAX_ITERATIONS) -> Dict:
        """Run weight optimization"""
        print("\n" + "="*70)
        print("WEIGHT OPTIMIZATION")
        print("="*70)

        meta_df = self.db.meta_df
        ground_truth = self.db.get_ground_truth()

        print(f"Ground truth records: {len(ground_truth):,}")

        # Initial MAPE
        initial_computed = self.compute_views(meta_df)
        initial_mape = self._calc_mape(initial_computed, meta_df['fc_uid'], ground_truth)
        print(f"Initial MAPE: {initial_mape:.2f}%")

        best_mape = initial_mape
        best_weights = self.current_weights.copy()

        for iteration in range(max_iterations):
            improved = False

            for category in ['genre_weights', 'type_weights', 'year_weights']:
                if category not in self.current_weights:
                    continue

                for key in list(self.current_weights[category].keys()):
                    original = self.current_weights[category][key]

                    # Try increase
                    self.current_weights[category][key] = original * (1 + self.learning_rate)
                    computed = self.compute_views(meta_df)
                    mape_up = self._calc_mape(computed, meta_df['fc_uid'], ground_truth)

                    # Try decrease
                    self.current_weights[category][key] = original * (1 - self.learning_rate)
                    computed = self.compute_views(meta_df)
                    mape_down = self._calc_mape(computed, meta_df['fc_uid'], ground_truth)

                    # Keep best
                    if mape_up < best_mape and mape_up < mape_down:
                        self.current_weights[category][key] = original * (1 + self.learning_rate)
                        best_mape = mape_up
                        best_weights = {k: dict(v) if isinstance(v, dict) else v
                                       for k, v in self.current_weights.items()}
                        improved = True
                    elif mape_down < best_mape:
                        self.current_weights[category][key] = original * (1 - self.learning_rate)
                        best_mape = mape_down
                        best_weights = {k: dict(v) if isinstance(v, dict) else v
                                       for k, v in self.current_weights.items()}
                        improved = True
                    else:
                        self.current_weights[category][key] = original

            self.mape_history.append(best_mape)

            # Check convergence
            if len(self.mape_history) > 1:
                improvement = self.mape_history[-2] - self.mape_history[-1]
                if improvement < CONVERGENCE_THRESHOLD:
                    print(f"Converged at iteration {iteration + 1}")
                    break

            # Decay learning rate
            if not improved:
                self.learning_rate *= LEARNING_RATE_DECAY
                if self.learning_rate < MIN_LEARNING_RATE:
                    print(f"Learning rate minimum reached at iteration {iteration + 1}")
                    break

            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}: MAPE = {best_mape:.2f}%")

        self.current_weights = best_weights

        improvement = initial_mape - best_mape
        print(f"\nFinal MAPE: {best_mape:.2f}%")
        print(f"Improvement: {improvement:.2f}% ({improvement/initial_mape*100:.1f}% relative)")

        return {
            'initial_mape': initial_mape,
            'final_mape': best_mape,
            'improvement': improvement,
            'iterations': len(self.mape_history),
            'weights': best_weights
        }

    def _calc_mape(self, computed: pd.Series, id_col: pd.Series, ground_truth: Dict) -> float:
        """Quick MAPE calculation"""
        total_ape = 0.0
        matched = 0

        for idx, comp in computed.items():
            fc_uid = str(id_col.iloc[idx])
            if fc_uid in ground_truth:
                actual = ground_truth[fc_uid]
                if actual > 0 and not pd.isna(comp) and comp > 0:
                    total_ape += abs(comp - actual) / actual
                    matched += 1

        return (total_ape / matched * 100) if matched > 0 else 100.0

    def save_weights(self):
        """Save optimized weights"""
        output = {
            'version': '28.00',
            'updated': datetime.now().isoformat(),
            'schema_version': '27.72',
            'weights': self.current_weights,
            'mape_history': self.mape_history[-10:]
        }

        opt_path = self.comp_dir / 'OPTIMIZED_WEIGHTS.json'
        safe_json_save(str(opt_path), output)
        print(f"Saved weights to {opt_path}")


# ============================================================================
# MAIN RUNNER
# ============================================================================
class MAPIE_V28_Runner:
    """Main MAPIE V28 runner for split database architecture"""

    def __init__(self):
        self.db_loader = SplitDatabaseLoader(BASE_DIR)
        self.mape_calc = None
        self.optimizer = None

    def run(self, optimize: bool = True) -> Dict:
        """Run full MAPIE examination cycle"""
        print("\n" + "="*70)
        print("MAPIE V28 - SPLIT DATABASE ARCHITECTURE")
        print("="*70)
        print(f"Timestamp: {datetime.now().isoformat()}")

        t_start = time.time()

        # Load databases
        self.db_loader.load_all()

        # Initialize calculators
        self.mape_calc = MAPECalculator(self.db_loader)
        self.optimizer = WeightOptimizer(self.db_loader)

        # Calculate current MAPE
        print("\n" + "="*70)
        print("CURRENT MAPE ANALYSIS")
        print("="*70)

        mape_result = self.mape_calc.calculate_mape('views_computed')
        if mape_result['status'] == 'success':
            print(f"views_computed MAPE: {mape_result['mape']:.2f}%")
            print(f"R² Score: {mape_result['r2']:.4f}")
            print(f"Matched records: {mape_result['matched']:,} / {mape_result['total_ground_truth']:,}")
            print(f"Valid range: {mape_result['within_valid_range']}")

        # Compare all predictions
        print("\n[Comparing all prediction columns...]")
        comparison = self.mape_calc.compare_predictions()
        print(f"\nBest model: {comparison['best_model']} (MAPE: {comparison['best_mape']:.2f}%)")
        for model, metrics in sorted(comparison['results'].items(), key=lambda x: x[1]['mape']):
            print(f"  {model:30} MAPE: {metrics['mape']:6.2f}%  R²: {metrics['r2']:.4f}")

        # Optimize weights
        opt_result = None
        if optimize:
            opt_result = self.optimizer.optimize()
            self.optimizer.save_weights()

        # Generate report
        runtime = time.time() - t_start

        report = {
            'timestamp': datetime.now().isoformat(),
            'version': '28.00',
            'schema_version': '27.72',
            'database_stats': self.db_loader.stats,
            'mape_analysis': mape_result if mape_result['status'] == 'success' else None,
            'model_comparison': comparison,
            'optimization': opt_result,
            'runtime_seconds': runtime,
            'status': 'SUCCESS'
        }

        # Save report
        report_path = Path(MAPIE_DIR) / f"MAPIE_V28_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        safe_json_save(str(report_path), report)
        print(f"\nReport saved: {report_path}")

        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Database rows: {self.db_loader.stats['meta_rows']:,}")
        if mape_result['status'] == 'success':
            print(f"Current MAPE: {mape_result['mape']:.2f}%")
        if opt_result:
            print(f"Optimized MAPE: {opt_result['final_mape']:.2f}%")
            print(f"Improvement: {opt_result['improvement']:.2f}%")
        print(f"Runtime: {fmt(runtime)}")

        return report


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='MAPIE V28 - Split Database Architecture')
    parser.add_argument('--no-optimize', action='store_true', help='Skip weight optimization')
    parser.add_argument('--analyze-only', action='store_true', help='Only run MAPE analysis')

    args = parser.parse_args()

    runner = MAPIE_V28_Runner()
    result = runner.run(optimize=not args.no_optimize and not args.analyze_only)

    return result


if __name__ == '__main__':
    main()
