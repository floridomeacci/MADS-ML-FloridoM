# Reddit Post Popularity Predictor

Predict whether a Reddit post about clothing will be popular based on its title, body text, and metadata.

## Dataset
- **24,548 Reddit posts** about "Clothing"
- Features: title, body, score, num_comments, subreddit, timestamps
- Target: Binary classification (popular vs unpopular based on median score)

## Model Architecture
**Neural Network with 4 layers:**
- Input layer: 512 features (12 numeric + 500 TF-IDF text features)
- Hidden layer 1: 256 neurons + ReLU + BatchNorm + Dropout(0.3)
- Hidden layer 2: 128 neurons + ReLU + BatchNorm + Dropout(0.3)
- Hidden layer 3: 64 neurons + ReLU + Dropout(0.3)
- Output layer: 1 neuron + Sigmoid (binary classification)

## Features Engineered
### Numeric Features (12):
- `text_length`, `title_length`, `body_length`, `word_count`
- `has_body`, `title_uppercase_ratio`
- `has_question_mark`, `has_exclamation`
- `num_comments_log`, `is_video`, `is_self`, `has_flair`

### Text Features (500):
- TF-IDF vectors from title + body text
- Stop words removed, min document frequency = 5

## Training Setup
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)
- **Loss:** Binary Cross Entropy
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)
- **Batch size:** 128
- **Epochs:** 50
- **Train/Test split:** 80/20

## Files
- `train_popularity.py` - Main training script
- `predict.py` - Make predictions on new posts
- `reddit_dataset.json` - Original dataset
- `popularity_model.pt` - Trained model checkpoint
- `training_curves.png` - Loss and accuracy over epochs
- `feature_importance.png` - Top 20 most important features
- `confusion_matrix.png` - Model performance visualization

## Usage

### Train the model:
```bash
cd side_project
python train_popularity.py
```

### Make predictions:
```python
python predict.py
```

## Expected Results
- **Accuracy:** ~70-75% (predicting popularity is challenging!)
- **Baseline:** 50% (random guessing)
- Model learns that:
  - Text length matters
  - Number of comments correlates with popularity
  - Certain words/phrases indicate popular content
  - Title characteristics affect engagement

## Next Steps
- Try LSTM for sequential text processing
- Add subreddit as categorical feature
- Predict actual score (regression) instead of binary
- Add temporal features (day of week, hour posted)
- Ensemble multiple models
