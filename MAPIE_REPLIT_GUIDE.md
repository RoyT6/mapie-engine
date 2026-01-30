# MAPIE: ML Guide for Replit

## Quick Start

MAPIE is a viewership prediction system using XGBoost + CatBoost ensemble.

### Installation (Replit)

```python
# In your replit.nix or requirements.txt:
pip install xgboost catboost pandas numpy scikit-learn pyarrow
```

### Loading the Data

```python
import pandas as pd

# Load the Fact Table (titles + predictions)
bfd = pd.read_parquet('Cranberry_BFD_V4.15.parquet')
print(f"BFD: {len(bfd):,} titles x {len(bfd.columns)} columns")

# Load the Dimension Table (country x platform views)
star = pd.read_parquet('Cranberry_Star_Schema_V4.15.parquet')
print(f"Star Schema: {len(star):,} rows")
```

### Key Columns in BFD

| Column | Type | Description |
|--------|------|-------------|
| `fc_uid` | string | Unique identifier (primary key) |
| `imdb_id` | string | IMDb ID (tt1234567 format) |
| `title` | string | Title name |
| `views_y` | int64 | **Predicted total views** |
| `abs_social_buzz` | float | Social media signal (0-100) |
| `abs_trend_velocity` | float | Trending momentum |
| `imdb_rating` | float | IMDb rating (0-10) |
| `tmdb_popularity` | float | TMDB popularity score |

### Key Columns in Star Schema

| Column | Type | Description |
|--------|------|-------------|
| `fc_uid` | string | Foreign key to BFD |
| `country` | string | Country code (US, GB, IN, etc.) |
| `platform` | string | Platform (netflix, prime, disney, etc.) |
| `views` | int64 | Views for this country+platform |

### Basic Queries

```python
# Top 10 most viewed titles
top_titles = bfd.nlargest(10, 'views_y')[['title', 'views_y']]
print(top_titles)

# Views by country
country_views = star.groupby('country')['views'].sum().sort_values(ascending=False)
print(country_views)

# Views by platform
platform_views = star.groupby('platform')['views'].sum().sort_values(ascending=False)
print(platform_views)

# Netflix US top titles
netflix_us = star[(star['platform'] == 'netflix') & (star['country'] == 'US')]
netflix_us_top = netflix_us.merge(bfd[['fc_uid', 'title']], on='fc_uid')
print(netflix_us_top.nlargest(10, 'views')[['title', 'views']])
```

### Training Your Own Model

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

# Load BFD
bfd = pd.read_parquet('Cranberry_BFD_V4.15.parquet')

# Define features (NO views columns!)
FORBIDDEN = ['views', 'hours_viewed', 'target', 'watched', 'viewership']
features = [c for c in bfd.columns
            if bfd[c].dtype in ['float64', 'int64', 'float32']
            and not any(f in c.lower() for f in FORBIDDEN)]

# Prepare data
X = bfd[features].fillna(0)
y = np.log1p(bfd['views_y'])  # Log transform

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train XGBoost
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.05
)
model.fit(X_train, y_train)

# Predict
y_pred = np.expm1(model.predict(X_test))  # Back to original scale
```

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAPIE PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Training Data ──────┐                                          │
│  (Netflix views)     │                                          │
│                      ▼                                          │
│  BFD Features ──► [XGBoost + CatBoost] ──► Global Predictions  │
│  (233 columns)       │ GPU Ensemble │                           │
│                      └──────────────┘                           │
│                             │                                   │
│                             ▼                                   │
│                      ┌─────────────┐                            │
│                      │ Calibration │                            │
│                      └──────┬──────┘                            │
│                             │                                   │
│              ┌──────────────┼──────────────┐                    │
│              ▼              ▼              ▼                    │
│         ┌────────┐    ┌──────────┐   ┌──────────┐              │
│         │  BFD   │    │ Country  │   │ Platform │              │
│         │ V4.15  │    │ Weights  │   │ Weights  │              │
│         └────┬───┘    └────┬─────┘   └────┬─────┘              │
│              │             │              │                     │
│              └─────────────┼──────────────┘                     │
│                            ▼                                    │
│                    ┌─────────────┐                              │
│                    │ Star Schema │                              │
│                    │    V4.15    │                              │
│                    └─────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Countries (18)

US, CN, IN, GB, BR, DE, JP, FR, CA, MX, AU, ES, IT, KR, NL, SE, SG, ROW

### Platforms (15)

netflix, prime, hulu, disney, hbo, peacock, apple, paramount, starz,
discovery, tubi, plutotv, britbox, mubi, curiosity

### Performance Stats

- MAPE: 34.60% (mean), 21.79% (median)
- R²: 0.62
- BFD: 768,641 titles
- Star Schema: 207M rows
- Total Views: 6.18 trillion

### Tips for Replit

1. **Memory**: Star Schema is 583MB - use chunked reading if needed
2. **Speed**: Filter BFD first, then join to Star Schema
3. **GPU**: Replit doesn't have GPU - use smaller n_estimators (100 vs 300)
4. **Storage**: Consider keeping only columns you need

```python
# Memory-efficient loading
columns_needed = ['fc_uid', 'title', 'views_y', 'imdb_rating']
bfd_small = pd.read_parquet('Cranberry_BFD_V4.15.parquet', columns=columns_needed)
```
