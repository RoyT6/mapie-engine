#!/usr/bin/env python3
"""
MAPIE AUTO-TRIGGER SYSTEM
=========================
VERSION: 1.0.0
CREATED: 2026-01-19

PURPOSE: Automatically triggers MAPIE examination when new twin database
versions are detected in the Downloads folder.

HOW IT WORKS:
1. Uses Windows/Linux file system watcher to monitor Downloads folder
2. Detects new Cranberry_BFD_V*.parquet and Cranberry_Star_Schema_V*.parquet files
3. When a matching pair (twin) is detected, triggers MAPIE_CONTINUAL_EXAMINATION
4. Runs weight optimization cycle
5. Updates MAPIE tracking logs

INSTALLATION:
- Windows: Add to Task Scheduler to run at startup
- Linux: Add to systemd or crontab

USAGE:
  python MAPIE_AUTO_TRIGGER.py                    # Run in foreground
  python MAPIE_AUTO_TRIGGER.py --daemon           # Run as daemon
  python MAPIE_AUTO_TRIGGER.py --once             # Run once on latest version

============================================================================
"""
import os
import sys
import time
import json
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Dict
import threading
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/mnt/c/Users/RoyT6/Downloads/MAPIE/auto_trigger.log')
    ]
)
logger = logging.getLogger('MAPIE_AUTO_TRIGGER')

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = '/mnt/c/Users/RoyT6/Downloads'
MAPIE_DIR = f'{BASE_DIR}/MAPIE'
COMP_DIR = f'{BASE_DIR}/Components'

# Files to monitor
BFD_PATTERN = re.compile(r'Cranberry_BFD.*V(\d+\.\d+)\.parquet$')
STAR_PATTERN = re.compile(r'Cranberry_Star_Schema.*V(\d+\.\d+)\.parquet$')

# Check interval (seconds)
DEFAULT_POLL_INTERVAL = 30

# Grace period after file creation before processing (seconds)
# Allows file to finish writing
GRACE_PERIOD = 10


# ============================================================================
# STATE MANAGEMENT
# ============================================================================
class TriggerState:
    """Manages the state of processed versions"""

    def __init__(self, state_file: str):
        self.state_file = state_file
        self.processed_versions: Set[str] = set()
        self.pending_twins: Dict[str, Dict] = {}  # version -> {bfd: path, star: path, detected_at: time}
        self._load()

    def _load(self):
        """Load state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.processed_versions = set(data.get('processed_versions', []))
                    logger.info(f"Loaded state: {len(self.processed_versions)} processed versions")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")

    def save(self):
        """Save state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    'processed_versions': list(self.processed_versions),
                    'updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save state: {e}")

    def is_processed(self, version: str) -> bool:
        """Check if version has been processed"""
        return version in self.processed_versions

    def mark_processed(self, version: str):
        """Mark version as processed"""
        self.processed_versions.add(version)
        self.save()

    def register_file(self, version: str, file_type: str, path: str):
        """Register a detected file"""
        if version not in self.pending_twins:
            self.pending_twins[version] = {
                'bfd': None,
                'star': None,
                'detected_at': time.time()
            }
        self.pending_twins[version][file_type] = path
        self.pending_twins[version]['detected_at'] = time.time()

    def get_ready_twins(self) -> list:
        """Get versions that have both BFD and Star Schema files ready"""
        ready = []
        current_time = time.time()

        for version, files in list(self.pending_twins.items()):
            # Check if both files exist
            if files['bfd'] and files['star']:
                # Check if grace period has passed
                if current_time - files['detected_at'] >= GRACE_PERIOD:
                    # Verify files still exist and are readable
                    if os.path.exists(files['bfd']) and os.path.exists(files['star']):
                        ready.append((version, files['bfd'], files['star']))
                        del self.pending_twins[version]

        return ready


# ============================================================================
# FILE WATCHER
# ============================================================================
class DatabaseWatcher:
    """Watches for new database files in the Downloads folder"""

    def __init__(self, base_dir: str, state: TriggerState):
        self.base_dir = Path(base_dir)
        self.state = state
        self.known_files: Dict[str, float] = {}  # path -> mtime
        self._initial_scan()

    def _initial_scan(self):
        """Scan existing files to establish baseline"""
        for f in self.base_dir.glob('Cranberry_*.parquet'):
            self.known_files[str(f)] = f.stat().st_mtime

        logger.info(f"Initial scan: {len(self.known_files)} parquet files")

    def check_for_new_files(self) -> list:
        """Check for new or modified database files"""
        new_versions = []

        for f in self.base_dir.glob('Cranberry_*.parquet'):
            fpath = str(f)
            mtime = f.stat().st_mtime

            # Check if this is a new or modified file
            if fpath not in self.known_files or self.known_files[fpath] != mtime:
                self.known_files[fpath] = mtime

                # Extract version
                bfd_match = BFD_PATTERN.search(f.name)
                star_match = STAR_PATTERN.search(f.name)

                if bfd_match:
                    version = bfd_match.group(1)
                    if not self.state.is_processed(version):
                        self.state.register_file(version, 'bfd', fpath)
                        logger.info(f"Detected BFD V{version}: {f.name}")

                elif star_match:
                    version = star_match.group(1)
                    if not self.state.is_processed(version):
                        self.state.register_file(version, 'star', fpath)
                        logger.info(f"Detected Star Schema V{version}: {f.name}")

        # Check for ready twin pairs
        return self.state.get_ready_twins()


# ============================================================================
# MAPIE RUNNER
# ============================================================================
class MAPIERunner:
    """Runs MAPIE examination on detected databases"""

    def __init__(self):
        self.examination_script = f'{MAPIE_DIR}/MAPIE_CONTINUAL_EXAMINATION.py'
        self.running = False

    def run_examination(self, version: str, bfd_path: str, star_path: str) -> bool:
        """Run MAPIE examination on a version"""
        if self.running:
            logger.warning("Examination already running, skipping")
            return False

        self.running = True
        logger.info(f"Starting MAPIE examination for V{version}")

        try:
            # Import and run the examination engine
            sys.path.insert(0, MAPIE_DIR)

            from MAPIE_CONTINUAL_EXAMINATION import ContinualExaminationEngine

            engine = ContinualExaminationEngine()
            result = engine.run_examination(version)

            if result.get('status') == 'SUCCESS':
                logger.info(f"Examination complete for V{version}")
                logger.info(f"  Final MAPE: {result['metrics']['final_mape']:.2f}%")
                logger.info(f"  Improvement: {result['metrics']['improvement']:.2f}%")
                return True
            else:
                logger.error(f"Examination failed: {result.get('message', 'Unknown error')}")
                return False

        except Exception as e:
            logger.error(f"Examination error: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.running = False

    def run_directly(self, version: str):
        """Run examination using direct imports (for when module is already loaded)"""
        if self.running:
            logger.warning("Examination already running, skipping")
            return False

        self.running = True

        try:
            # Run examination directly
            os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
            os.environ['NUMBA_CUDA_USE_NVIDIA_BINDING'] = '1'
            os.environ['CUDF_SPILL'] = 'on'

            # Import examination module
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "examination",
                self.examination_script
            )
            examination = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(examination)

            # Run examination
            engine = examination.ContinualExaminationEngine()
            result = engine.run_examination(version)

            return result.get('status') == 'SUCCESS'

        except Exception as e:
            logger.error(f"Direct examination error: {e}")
            return False

        finally:
            self.running = False


# ============================================================================
# MAIN TRIGGER LOOP
# ============================================================================
class AutoTrigger:
    """Main auto-trigger service"""

    def __init__(self):
        self.state = TriggerState(f'{MAPIE_DIR}/.trigger_state.json')
        self.watcher = DatabaseWatcher(BASE_DIR, self.state)
        self.runner = MAPIERunner()
        self.should_stop = False

    def run_once(self):
        """Run examination on the latest version"""
        logger.info("Running single examination on latest version")

        # Find latest version
        latest_version = None
        latest_mtime = 0

        for f in Path(BASE_DIR).glob('Cranberry_BFD_*.parquet'):
            match = BFD_PATTERN.search(f.name)
            if match:
                version = match.group(1)
                mtime = f.stat().st_mtime
                if mtime > latest_mtime:
                    latest_version = version
                    latest_mtime = mtime

        if latest_version:
            bfd_path = str(next(Path(BASE_DIR).glob(f'Cranberry_BFD_*V{latest_version}.parquet')))
            star_files = list(Path(BASE_DIR).glob(f'Cranberry_Star_Schema_*V{latest_version}.parquet'))
            star_path = str(star_files[0]) if star_files else None

            if star_path:
                success = self.runner.run_examination(latest_version, bfd_path, star_path)
                if success:
                    self.state.mark_processed(latest_version)
            else:
                logger.warning(f"No Star Schema found for V{latest_version}")
        else:
            logger.warning("No database versions found")

    def run_daemon(self, poll_interval: int = DEFAULT_POLL_INTERVAL):
        """Run as continuous daemon"""
        logger.info(f"Starting auto-trigger daemon")
        logger.info(f"Monitoring: {BASE_DIR}")
        logger.info(f"Poll interval: {poll_interval} seconds")

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        while not self.should_stop:
            try:
                # Check for new files
                ready_twins = self.watcher.check_for_new_files()

                for version, bfd_path, star_path in ready_twins:
                    logger.info(f"Twin databases ready for V{version}")

                    success = self.runner.run_examination(version, bfd_path, star_path)

                    if success:
                        self.state.mark_processed(version)
                        logger.info(f"V{version} processed successfully")
                    else:
                        logger.warning(f"V{version} processing failed")

                # Sleep
                time.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Daemon error: {e}")
                time.sleep(poll_interval)

        logger.info("Auto-trigger daemon stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.should_stop = True


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='MAPIE Auto-Trigger System')
    parser.add_argument('--daemon', '-d', action='store_true', help='Run as daemon')
    parser.add_argument('--once', '-o', action='store_true', help='Run once on latest version')
    parser.add_argument('--interval', '-i', type=int, default=30, help='Poll interval (seconds)')

    args = parser.parse_args()

    trigger = AutoTrigger()

    if args.once:
        trigger.run_once()
    elif args.daemon:
        trigger.run_daemon(args.interval)
    else:
        # Default: run in foreground with logging
        print("MAPIE Auto-Trigger")
        print("==================")
        print("Use --daemon for background mode or --once for single run")
        print("\nRunning single examination...")
        trigger.run_once()


if __name__ == '__main__':
    main()
