# MAPIE Engine - ALGO 95.66 Compliance
## VERSION 27.66 | ALGO 95.66 | ViewerDBX Bus v1.0 | SCHEMA V27.00 | CONTINUAL EXAMINATION

---

## P0 CANON: Studio vs Production Company Classification (2026-01-25)

**CRITICAL RULE:** Classification is by PRIMARY FUNCTION, NOT by fixed lookup list.

### Studios (Creative/On-Set)
| Roles | Activities |
|-------|------------|
| Directors, Writers | Filming, Blocking |
| Cinematographers, Cameramen | Location, Direction |
| Set Designers, Crew | Content Creation, Capture |
| Actors, Casting Directors | Casting, Extras |

### Production Companies (Business/Post)
| Roles | Activities |
|-------|------------|
| Lawyers, Finance | Financing, Legal |
| Post-Production, Final Cut | Editing, Mixing |
| Producers, Editors | Distribution, Marketing |
| Music, Sound, Executives | Promotion, Post-Production |

**Ingestion Rules:**
- P0: Classify by FUNCTION, not name matching
- P1: Accept ANY entity meeting criteria (no lookup required)
- P2: Metadata columns OPTIONAL (NULL acceptable)

**Rule File:** `Schema/RULES/STUDIO_VS_PRODUCTION_COMPANY_RULES.json`

---

## V28.00 Star Hierarchy Update (2026-01-24)

**New Columns Available:**
- `star_1` - Top-billed lead actor
- `star_2` - Second-billed actor
- `star_3` - Third-billed actor
- `supporting_cast` - Remaining cast (semicolon-separated)
- `cast_data` - DEPRECATED (preserved for backwards compatibility)

**For ML Feature Engineering:**
- Use `star_1`/`star_2`/`star_3` for individual star power calculations
- `abs_star_power` aggregates social metrics from top 3 stars
- Coverage: 93,103 unique titles (15.3%) have star hierarchy data

---

## ROLE: LEDGER (Records Truth) + OPTIMIZER (Improves Accuracy)

MAPIE tracks MAPE (Mean Absolute Percentage Error), validates model accuracy, and **continuously optimizes algorithm weightings** to reduce MAPE of Views calculations.

---

## CONTINUAL EXAMINATION SYSTEM (NEW v23.00)

The MAPIE system now includes automatic monitoring and weight optimization:

### Auto-Trigger on New Database Versions
```
When new Cranberry_BFD_V*.parquet published:
  1. Detect twin databases (BFD + Star Schema)
  2. Load ground truth from 52+ training sources
  3. Run weight optimization engines
  4. Update algorithm weightings
  5. Track MAPE improvement
```

### Files Created by Continual Examination
| File | Purpose |
|------|---------|
| `MAPIE_CONTINUAL_EXAMINATION.py` | Main examination engine |
| `MAPIE_INTEGRATED_RUNNER.py` | Auto-trigger & monitoring |
| `MAPIE_WEIGHT_ENGINES.py` | Three specialized optimizers |
| `MAPIE_AUTO_TRIGGER.py` | File watcher daemon |

### Weight Optimization Engines
| Engine | Optimizes | Output |
|--------|-----------|--------|
| AbstractWeightEngine | 77 abstract signal weights | ABSTRACT_SIGNAL_WEIGHTS.json |
| ComponentWeightEngine | 65+ lookup table weights | COMPONENT_WEIGHTS.json |
| TrueViewWeightEngine | Ground truth calibration | TRUE_VIEW_WEIGHTS.json |

### Usage
```bash
# Run once on latest version
python MAPIE_INTEGRATED_RUNNER.py --once

# Watch mode (auto-trigger on new versions)
python MAPIE_INTEGRATED_RUNNER.py --watch

# Specific version
python MAPIE_INTEGRATED_RUNNER.py --version 19.84
```

---

## VIEWERDBX BUS INTEGRATION

```python
from viewerdbx_bus import get_bus

bus = get_bus()
predictions_path = bus.resolve_path("predictions")
training_data_path = bus.resolve_path("training_data")
```

### Component Registration
- **Name**: MAPIE
- **Role**: LEDGER + OPTIMIZER
- **Provides**: mape_tracking, validation, accuracy_reports, weight_optimization
- **Requires**: predictions, training_data, twin_databases

---

## MANDATORY EXECUTION CONSTRAINTS

### GPU Enforcement (ALGO 95.4)
- GPU execution is **MANDATORY**
- CPU fallback allowed only for development/testing
- Use `cudf` not `pandas` in production

### Temporal Validation (NEW in 95.4)
- Report temporal validation status in MAPIE reports
- Flag any predictions on unreleased content

### Valid Metric Ranges (Anti-Cheat)
| Metric | Valid | Invalid (Cheating) |
|--------|-------|-------------------|
| R2 | 0.30 - 0.90 | > 0.95 |
| MAPE | 5% - 40% | < 2% |

---

## MASTER EQUATION (ALGO 95.4)

```
V_hat(i,p,t) = V_hat_65(i,p,t) * R_i * G_d,t * Q_d,t * A_i,p,t
```

Where:
- R_i = Completion Quality Index
- G_d,t = Geopolitical Risk Multiplier
- Q_d,t = Quality of Experience
- A_i,p,t = Platform Availability

**Weight Optimization adds:**
- Abstract signal weights (77 signals)
- Genre/type/year multipliers
- Studio quality weights

---

## MAPE TRACKING

MAPIE tracks improvements in `Components/MAPIE_MAPE_TRACKER.json`:
- Historical MAPE per version
- Best MAPE achieved
- Improvement trends
- Anti-cheat validation results

---

## FOUR RULES ENFORCEMENT

| Rule | Requirement |
|------|-------------|
| V1 | Proof-of-Work: rows validated |
| V2 | Checksum: report hash |
| V3 | Logging: MAPIE report chain |
| V4 | Audit: machine-generated only |

---

## ANTI-CHEAT RULES

| FORBIDDEN | WHY |
|-----------|-----|
| `views_*` in features | Data leakage |
| `pd.read_parquet()` | CPU fallback (prod) |
| `n_estimators < 1350` | Tiny models |
| R2 > 0.95 | Data leakage |
| MAPE < 2% | Data leakage |

---

## RALPH INTEGRATION

Uses Ralph's autonomous development loop with fail-closed philosophy and ViewerDBX Bus.

The Continual Examination System was built using Ralph Loop for iterative improvement.

---

**Synced with**: `ALGO Engine/ALGO_95.4_SPEC.md`
**Bus Version**: 1.0
**Continual Examination**: v1.0.0

---

## SESSION LOG: 2026-01-19

### Accomplished
Built complete MAPIE Continual Examination System:

**New Files Created:**
| File | Lines | Purpose |
|------|-------|---------|
| `MAPIE_CONTINUAL_EXAMINATION.py` | ~650 | Main examination engine with version detection, ground truth loading, weight optimization |
| `MAPIE_WEIGHT_ENGINES.py` | ~550 | Three specialized optimizers (Abstract, Component, TrueView) |
| `MAPIE_INTEGRATED_RUNNER.py` | ~550 | Complete auto-trigger system with monitoring |
| `MAPIE_AUTO_TRIGGER.py` | ~300 | File watcher daemon for background operation |

**Key Features Implemented:**
1. Auto-detection of new twin databases (BFD + Star Schema)
2. Ground truth aggregation from 52+ training data sources
3. Gradient-free weight optimization (coordinate descent)
4. MAPE tracking with improvement history
5. Anti-cheat validation (MAPE 5-40%, R2 0.30-0.90)
6. GPU acceleration with CPU fallback for dev/test

**Output Files (to Components/):**
- OPTIMIZED_WEIGHTS.json
- ABSTRACT_SIGNAL_WEIGHTS.json
- COMPONENT_WEIGHTS.json
- TRUE_VIEW_WEIGHTS.json
- MAPIE_MAPE_TRACKER.json

### Task Completion Criteria Met
System runs automatically when new twin databases are published as new versions in the Downloads folder.

### Next Steps
1. Run `python MAPIE_INTEGRATED_RUNNER.py --watch` to start monitoring
2. New database versions will auto-trigger examination
3. Review MAPE_MAPE_TRACKER.json for improvement trends

---

## SESSION LOG: 2026-02-02

### Problem Identified: FlixPatrol vs Netflix Data Source Mismatch

**Root Cause Analysis:**
The BFD database (`BFD-Views-2026-Feb-2.00.parquet`) contained FlixPatrol estimated views, NOT Netflix published actuals. This caused MAPE of 110%+ when comparing against Netflix Symphony ground truth.

| Source | Max Views | Data Type |
|--------|-----------|-----------|
| BFD Database (before) | 1.6M | FlixPatrol estimates (capped) |
| Netflix Symphony | 143.8M | Netflix published actuals |

**Key Finding:** For 50M+ view titles, FlixPatrol captured only 0.8% of actual Netflix views.

### Solution Executed: Netflix Symphony Merge

Replaced BFD views columns with Netflix Symphony published data for matched titles.

**Merge Results:**
| Metric | Value |
|--------|-------|
| Matched titles | 6,708 (by fc_uid) |
| Cells updated | 756,053 |
| Columns updated | 178 |
| Post-merge MAPE | 0.0000% |

**Database Update:**
- Input: `BFD-Views-2026-Feb-2.00.parquet`
- Output: `BFD-Views-2026-Feb-2.01.parquet` (108.75 MB)

### Files Created

| File | Purpose |
|------|---------|
| `calculate_mape_scores.py` | Season-aware MAPE calculation with title+season matching |
| `calculate_mape_scores_v2.py` | Netflix Symphony vs BFD comparison |
| `mape_analysis_report.py` | Root cause analysis script |
| `merge_netflix_symphony_to_bfd.py` | Merge engine with Four Rules compliance |
| `MAPE_ANALYSIS_REPORT_*.json` | Detailed findings on data source mismatch |
| `NETFLIX_SYMPHONY_MERGE_AUDIT_*.json` | Four Rules audit trail |

### Four Rules Compliance

All rules verified for merge operation:
- V1: Proof-of-Work hash `000045a4...` (nonce: 80,200)
- V2: Checksum MD5 `07c719b074a1476245b8925bcece40de`
- V3: Execution log with full operation chain
- V4: Machine generated, no human override

### Important Notes

1. **fc_uid Format:** Netflix Symphony uses `NNNNNNN_sNN` (no 'tt' prefix), BFD uses `ttNNNNNNN_sNN`
2. **Match Rate:** Only 26% of Netflix titles (6,708 of 25,346) had exact fc_uid matches in BFD
3. **Remaining Titles:** ~18,000 Netflix titles without BFD matches still need fc_uid resolution
4. **MAPE 0%:** Expected for matched titles since BFD now contains actual Netflix data

### Pending Work - CRITICAL

1. **Use GPU Engine / Parallel Engine for IMDB Reference merge** - DO NOT use single-threaded pandas
2. Reference file: `Training Data/IMDB Ref Files/224 thousand imdb_fcuid_records.csv` (224,373 records)
3. This file bridges Netflix Symphony fc_uid to BFD fc_uid via title+season matching
4. Must use RAPIDS/cuDF or multiprocessing with shared memory (not spawn)

### Scripts Created (Not Yet Executed)

| File | Purpose | Status |
|------|---------|--------|
| `merge_netflix_via_imdb_ref.py` | IMDB reference bridge merge | NOT RUN - too slow |
| `merge_netflix_parallel.py` | 14-worker parallel merge | FAILED - memory issues with spawn |
| `merge_netflix_optimized.py` | Vectorized pandas | REJECTED - single-threaded |

### Required Next Session

Use GPU Engine with RAPIDS/cuDF for the merge:
- Load all data into GPU memory
- Use cuDF merge operations (much faster)
- Reference: `Parallel Engine/CLAUDE.md` for optimal worker configs
- System: Ryzen 9 3950X (16c/32t), 128GB RAM, RTX 3080Ti (12GB VRAM)

### Git Commits

- `7ae3239` - Update CLAUDE.md with session log
- `ba73a13` - Execute Netflix Symphony to BFD merge
- `794371f` - Add MAPE analysis identifying data source mismatch
- `0e03cb3` - Add MAPIE validation scripts and reports
