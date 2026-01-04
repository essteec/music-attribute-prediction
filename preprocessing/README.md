# Preprocessing Pipeline

## Overview

This directory contains a modular, cache-aware preprocessing pipeline for the music prediction project. Each preprocessing step can be run independently and will automatically skip processing if outputs are up-to-date.

## Architecture

### Modular Design

The pipeline is split into independent modules:

- **`process_audio.py`** - Audio features (scaling, encoding, genre one-hot)
- **`process_text_stats.py`** - Text statistics (word counts, vocabulary metrics)
- **`process_sentiment.py`** - Sentiment analysis (TextBlob polarity & subjectivity)
- **`process_targets.py`** - Target variables (valence, energy, danceability, popularity)

### Intelligent Caching

The `pipeline_utils.py` module provides:

- **Hash-based change detection** - Tracks MD5 hashes of input files
- **Timestamp tracking** - Records when each step was last run
- **Dependency resolution** - Ensures dependent steps re-run when needed
- **Skip logic** - Automatically skips steps with up-to-date outputs

### Benefits

✅ **No redundant work** - Change audio features without re-running TextBlob  
✅ **Fast iteration** - Only reprocess what changed  
✅ **Transparent** - Clear logging of what's cached vs. what's running  
✅ **Modular** - Each step can run standalone for testing  

## Usage

### Basic Usage

Run all preprocessing steps (skips cached steps automatically):

```bash
cd preprocessing
python run_preprocessing.py
```

### Run Specific Steps

Only run audio and text statistics (skip sentiment):

```bash
python run_preprocessing.py --steps audio text_stats
```

Available steps: `audio`, `text_stats`, `sentiment`, `targets`

### Force Re-run

Ignore cache and force re-processing:

```bash
python run_preprocessing.py --force
```

Or force specific steps:

```bash
python run_preprocessing.py --steps sentiment --force
```

### Check Cache Status

See what's cached and when it was last run:

```bash
python run_preprocessing.py --status
```

### Clear Cache

Clear cache for all steps:

```bash
python run_preprocessing.py --clear
```

Clear cache for specific steps:

```bash
python run_preprocessing.py --clear audio sentiment
```

## Individual Modules

Each module can also be run standalone:

```bash
# Run just audio processing
python process_audio.py

# Run just text statistics
python process_text_stats.py

# Run just sentiment extraction
python process_sentiment.py

# Run just target transformation
python process_targets.py
```

## How Caching Works

### Change Detection

The pipeline tracks:

1. **Input file hashes** - MD5 hash of each CSV file
2. **Output file existence** - Check if output files exist
3. **Timestamps** - When each step was last completed
4. **Dependencies** - If prerequisite steps have been updated

### When Steps Run

A step will run if:

- Any output file is missing
- Any input file has changed (different hash)
- The step has never been run before
- A dependency was updated after this step

### When Steps Skip

A step will skip if:

- All output files exist
- All input files unchanged since last run
- No dependencies updated since last run

### Cache Storage

Cache metadata is stored in:

```
ml/features/.preprocessing_cache.json
```

This file tracks hashes, timestamps, and metadata for each step.

## File Organization

### Old Files (Deprecated)

These files are now replaced by the modular pipeline:

- ~~`audio_features.py`~~ → Use `process_audio.py`
- ~~`text_statistics.py`~~ → Use `process_text_stats.py`
- ~~`sentiment_features.py`~~ → Use `process_sentiment.py`
- ~~`apply_transformations.py`~~ → Use `run_preprocessing.py`

### New Modular Structure

```
preprocessing/
├── run_preprocessing.py      # Main orchestrator script
├── pipeline_utils.py          # Caching utilities
├── process_audio.py           # Audio feature processing
├── process_text_stats.py      # Text statistics extraction
├── process_sentiment.py       # Sentiment analysis
├── process_targets.py         # Target variable processing
├── data_splitting.py          # Artist-aware train/val/test splits
└── README.md                  # This file
```

## Output Files

All processed features are saved to `ml/features/`:

### Feature Arrays

```
features/
├── X_train_audio.npy          # Audio features (train)
├── X_val_audio.npy            # Audio features (val)
├── X_test_audio.npy           # Audio features (test)
├── X_train_text_stats.npy     # Text statistics (train)
├── X_val_text_stats.npy       # Text statistics (val)
├── X_test_text_stats.npy      # Text statistics (test)
├── X_train_sentiment.npy      # Sentiment features (train)
├── X_val_sentiment.npy        # Sentiment features (val)
├── X_test_sentiment.npy       # Sentiment features (test)
├── y_train_valence.npy        # Target: valence (train)
├── y_val_valence.npy          # Target: valence (val)
├── y_test_valence.npy         # Target: valence (test)
├── ... (similar for energy, danceability, popularity)
```

### Transformers and Metadata

```
features/
├── audio_scaler.pkl           # StandardScaler for audio continuous features
├── genre_encoder.pkl          # OneHotEncoder for genre
├── text_stats_scaler.pkl      # StandardScaler for text statistics
├── sentiment_scaler.pkl       # StandardScaler for sentiment
├── audio_feature_names.txt    # List of audio feature names
├── preprocessing_metadata.json # (deprecated, use cache instead)
└── .preprocessing_cache.json  # Cache metadata (hashes, timestamps)
```

## Troubleshooting

### Cache not working?

Check cache status:
```bash
python run_preprocessing.py --status
```

### Force re-run a step

```bash
python run_preprocessing.py --steps audio --force
```

### Clear stale cache

```bash
python run_preprocessing.py --clear
```

### Outputs exist but cache says missing?

Manually mark as complete by running once:
```bash
python run_preprocessing.py --force
```

## Next Steps

After preprocessing, features are ready for modeling:

```bash
cd ../models
python baseline_models.py          # Train baseline models
python text_stats_models.py        # Experiment with text features
python combined_text_models.py     # Combined audio + text models
```
