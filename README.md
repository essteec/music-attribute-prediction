# Music Attribute Prediction

A machine learning project to predict Spotify track attributes (**Valence**, **Energy**, **Danceability**, and **Popularity**) using a multi-modal approach combining audio features, lyrical analysis, and artist metadata.

## 🚀 Quick Start

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/essteec/music-attribute-prediction.git
cd music-attribute-prediction
python -m venv .venv

# Windows: 
.venv\Scripts\activate
# Linux/MacOS:
source .venv/bin/activate  

pip install -r requirements.txt
```

### 2. Data Preparation
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres).
2. Place the raw CSV files into the `data/` directory.
3. Run the data splitting script:
```bash
cd preprocessing
python data_splitting.py
```

### 3. Preprocessing Pipeline
Run the modular preprocessing pipeline to extract and transform features:
```bash
cd preprocessing
python run_preprocessing.py
```
*This handles audio scaling, sentiment analysis, text statistics, and semantic embeddings with intelligent caching.*

### 4. Training & Evaluation
Train baseline or enhanced models:
```bash
cd models
python baseline_models.py      # Train baseline models or any file ends with _models.py

python test_evaluation_final.py # Final evaluation on test set
```

## Project Structure

- `data/`: Raw dataset and train/val/test splits.
- `preprocessing/`: Modular scripts for feature engineering and NLP.
- `features/`: Processed feature arrays and saved transformers.
- `models/`: Model training, hyperparameter tuning, and evaluation scripts.
- `notebooks/`: Exploratory Data Analysis (EDA) and results visualization.
- `results/`: Performance metrics, figures, and reports.

## Features & Methodology

- **Audio:** 23 features including power-transformed acousticness, cyclical key encoding, and artist popularity.
- **Lyrics:** 
  - **Text Stats:** Word counts, vocabulary metrics, and complexity.
  - **Sentiment:** TextBlob polarity and subjectivity scores.
  - **Embeddings:** 384-dimensional semantic vectors using `all-MiniLM-L6-v2`.
- **Models:** Implementation of XGBoost, CatBoost, LightGBM, and Random Forest with Recursive Feature Elimination (RFE) for optimization.

## Results
The project evaluates models across four targets, comparing "Enhanced" (full 414 features) vs "RFE" (optimized subsets) configurations. Detailed metrics and plots can be found in the `results/` directory after running evaluations.
