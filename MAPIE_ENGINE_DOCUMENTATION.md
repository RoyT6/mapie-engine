# MAPIE - MAPE Improvement Engine
## Version 18.00 | Daily Runner Architecture
## Created: 2026-01-13 | Updated: 2026-01-14

---

# Overview

MAPIE (MAPE Improvement Engine) is a standalone daily runner that refreshes and updates the Cranberry database with computed viewership values.

## Key Characteristics
- **Standalone**: Runs independently after each algo cycle
- **Daily Schedule**: Designed to run once per day
- **Version Control**: Auto-increments version (V4.11 → V4.12 → V4.13)
- **Dual Output**: Updates both BFD (fact) and Star Schema (dimension) tables
- **Audit Trail**: Creates run logs for tracking

---

# Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MAPIE DAILY RUNNER                                  │
│                     (MAPIE_DAILY_RUNNER.py)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────┐    ┌──────────────────────────────────────────┐   │
│  │  Cranberry_BFD       │    │  Components/ (65+ JSON files)            │   │
│  │  V{current}.parquet  │    ├──────────────────────────────────────────┤   │
│  │  768K rows × 1702    │    │  • country_viewership_weights_2025.json  │   │
│  │  cols                │    │  • platform_allocation_weights.json      │   │
│  └──────────────────────┘    │  • cranberry genre decay table.json      │   │
│                              │  • Apply studio weighting.json            │   │
│                              │  • streaming_lookup_*.json (18 files)     │   │
│                              │  • TRAINING_MATRIX_UNIFIED.parquet        │   │
│                              └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PROCESSING LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 1: Load Current BFD                                           │    │
│  │  • Read parquet file                                                │    │
│  │  • Store original views_computed for comparison                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 2: Load Component Lookup Tables                               │    │
│  │  • Country weights (18 countries)                                   │    │
│  │  • Platform weights by country (48 platforms × 18 countries)        │    │
│  │  • Genre-platform affinity (21 genres)                              │    │
│  │  • Training statistics from Engine 1                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 3: Compute New views_computed                                 │    │
│  │                                                                     │    │
│  │  FORMULA:                                                           │    │
│  │  views_computed = base_views × genre_mult × type_mult × year_mult   │    │
│  │                                                                     │    │
│  │  WHERE:                                                             │    │
│  │    base_views = f(abstract_signals, training_distribution)          │    │
│  │    genre_mult = genre_decay_table[genre].halflife_factor            │    │
│  │    type_mult  = 1.2 (movie) | 0.9 (series)                          │    │
│  │    year_mult  = 1.3 (current) → 0.8 (>10 years)                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 4: Regenerate Star Schema                                     │    │
│  │                                                                     │    │
│  │  FOR EACH title:                                                    │    │
│  │    FOR EACH country (18):                                           │    │
│  │      country_views = views_computed × country_weight                │    │
│  │      FOR EACH platform (varies by country):                         │    │
│  │        platform_views = country_views × platform_share × affinity   │    │
│  │        → Write row (fc_uid, country, platform, views)               │    │
│  │                                                                     │    │
│  │  OUTPUT: ~118M rows (768K × 18 × ~8 platforms avg)                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Cranberry_BFD_MAPIE_RUN_{date}_V{version}.parquet                   │   │
│  │  • 768,641 rows × 1,702 columns                                      │   │
│  │  • ~2.7 GB                                                           │   │
│  │  • Updated views_computed column                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Cranberry_Star_Schema_MAPIE_RUN_{date}_V{version}.parquet           │   │
│  │  • ~118,000,000 rows × 4 columns                                     │   │
│  │  • ~220 MB                                                           │   │
│  │  • Columns: fc_uid, country, platform, views                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Components/MAPIE_RUN_LOG.json                                       │   │
│  │  • Audit trail with timestamps, versions, stats                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# Data Flow Diagram

```
                    ┌─────────────────────────────────────────┐
                    │        EXTERNAL DATA SOURCES            │
                    └─────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
          ▼                             ▼                             ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  ENGINE 1           │   │  ENGINE 2           │   │  Components/        │
│  Training Data      │   │  Abstract Data      │   │  Lookup Tables      │
│  Loader             │   │  Loader             │   │                     │
├─────────────────────┤   ├─────────────────────┤   ├─────────────────────┤
│  • 26 training files│   │  • 78 abstract files│   │  • 65+ JSON configs │
│  • 139K records     │   │  • 77 signals       │   │  • Streaming lookups│
│  • Views + metadata │   │  • Market data      │   │  • Country weights  │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
          │                             │                             │
          └─────────────────────────────┼─────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │           MAPIE DAILY RUNNER            │
                    │     (Runs once daily, auto-versions)    │
                    └─────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    ▼                                       ▼
        ┌─────────────────────┐               ┌─────────────────────┐
        │  Cranberry_BFD      │               │  Cranberry_Star_    │
        │  (Fact Table)       │               │  Schema (Dimension) │
        ├─────────────────────┤               ├─────────────────────┤
        │  768,641 rows       │               │  118M+ rows         │
        │  1,702 columns      │               │  4 columns          │
        │  views_computed     │──────────────▶│  fc_uid, country,   │
        │  updated            │   (derived)   │  platform, views    │
        └─────────────────────┘               └─────────────────────┘
```

---

# Version History

| Run Date | Version | BFD Rows | Star Rows | Total Views | Runtime | Notes |
|----------|---------|----------|-----------|-------------|---------|-------|
| 2026-01-14 | **V18.00** | 768,641 | 204,205,513 | 6,083,408,968,197 | 10.3m | **MAJOR VERSION** - Schema updates for start_year/end_year/max_seasons |
| 2026-01-14 | V4.17 | 768,641 | - | - | - | Release validation fix (start_year canon) |
| 2026-01-14 | V4.16 | 768,641 | 576,267,237 | - | - | Platform exclusivity detection |
| 2026-01-13 | V4.15 | 768,641 | 380,678,209 | - | - | Optimized GPU version |
| 2026-01-13 | V4.12 | 768,641 | 118,370,714 | 663,966,973,239 | 9m 19s | Initial MAPIE run |
| (previous) | V4.11 | 768,641 | 69,177,690 | 1,208,696,543,158 | - | Pre-MAPIE baseline |

---

# File Naming Convention

```
Cranberry_BFD_V{major}.{minor}.parquet
Cranberry_Star_Schema_V{major}.{minor}.parquet

Current (V18.00):
  Cranberry_BFD_V18.00.parquet
  Cranberry_Star_Schema_V18.00.parquet

Previous versions archived as:
  Cranberry_BFD_V4.17.parquet
  Cranberry_Star_Schema_V4.16.parquet
```

---

# Component Dependencies

## Required Files

| File | Purpose | Location |
|------|---------|----------|
| `Cranberry_BFD_V{current}.parquet` | Input database | Downloads/ |
| `country_viewership_weights_2025.json` | Country distribution | Components/ |
| `platform_allocation_weights.json` | Platform shares + affinity | Components/ |
| `TRAINING_DATA_STATS.json` | Training distribution | Components/ |
| `TRAINING_MATRIX_UNIFIED.parquet` | Training records (Engine 1) | Components/ |

## Supporting Engines

| Engine | File | Purpose |
|--------|------|---------|
| ENGINE 1 | `ENGINE_1_TRAINING_DATA_LOADER.py` | Load all training data |
| ENGINE 2 | `ENGINE_2_ABSTRACT_DATA_LOADER.py` | Load all abstract data |
| ENGINE 3 | `ENGINE_3_COMPONENT_VIEW_COMPUTER.py` | Compute views with lookups |
| MAPIE | `MAPIE_DAILY_RUNNER.py` | Daily refresh + versioning |

---

# Scheduling (Recommended)

## Linux/WSL (cron)
```bash
# Run daily at 2:00 AM
0 2 * * * /home/user/miniforge3/envs/rapids-24.12/bin/python3 /mnt/c/Users/RoyT6/Downloads/GPU\ Enablement/MAPIE_DAILY_RUNNER.py >> /var/log/mapie.log 2>&1
```

## Windows (Task Scheduler)
```
Action: Start a program
Program: wsl
Arguments: -- bash -c "source ~/miniforge3/etc/profile.d/conda.sh && conda activate rapids-24.12 && python3 '/mnt/c/Users/RoyT6/Downloads/GPU Enablement/MAPIE_DAILY_RUNNER.py'"
Trigger: Daily at 02:00
```

---

# Run Log Format

```json
{
  "run_timestamp": "2026-01-13T14:18:14.521708",
  "previous_version": "4.11",
  "new_version": "4.12",
  "input_file": "Cranberry_BFD_V4.11.parquet",
  "output_files": {
    "bfd": "Cranberry_BFD_MAPIE_RUN_20260113_V4.12.parquet",
    "star_schema": "Cranberry_Star_Schema_MAPIE_RUN_20260113_V4.12.parquet"
  },
  "stats": {
    "bfd_rows": 768641,
    "bfd_columns": 1702,
    "star_rows": 118370714,
    "views_computed_total": 663966973239,
    "views_computed_mean": 863819,
    "countries": 18,
    "platforms": 154
  },
  "runtime_seconds": 559.5
}
```

---

**Document Version**: 18.00
**Engine Version**: MAPIE V18.00
**Created**: 2026-01-13
**Updated**: 2026-01-14
**Author**: FC-ALGO-80 Pipeline

---

# V18.00 Schema Updates

## Key Changes in V18.00

| Field | Purpose | Source |
|-------|---------|--------|
| `start_year` | **CANONICAL** for release validation | IMDb startYear, TMDB air_date |
| `end_year` | Series conclusion detection | IMDb endYear, TMDB last_air_date |
| `max_seasons` | Season row integrity audit | TMDB number_of_seasons |

## Release Validation Rules (V18.00)

1. **Primary**: Check `status` field - if 'Upcoming', 'Announced', 'In Production' → block allocation
2. **Secondary**: Check `start_year > current_year` → block allocation
3. **DO NOT** use `premiere_date` for release validation (inherits from series)

## Final Season Detection (V18.00)

1. Series has ended if `end_year` is populated AND `end_year <= current_year`
2. Final season = `season_number == max_seasons` AND series has ended
3. For ended series, views decay should be accelerated

## Audits (V18.00)

| Audit | Check | Severity |
|-------|-------|----------|
| audit_1 | Row count matches max_seasons per series | CRITICAL |
| audit_2 | season_number sequential 1...max_seasons | CRITICAL |
| audit_4 | No season_number > max_seasons (orphans) | CRITICAL |

## V18.00 Statistics

- **BFD Rows**: 768,641
- **BFD Columns**: 1,708
- **Star Schema Rows**: 204,205,513
- **Exclusive Titles**: 4,798 (prime: 3,800, disney: 858, netflix: 140)
- **Unreleased Blocked**: 7,848 (1.02%)
- **Final Seasons**: 69,920
- **Series Ended**: 87,649
