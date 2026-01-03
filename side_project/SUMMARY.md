# Reddit Post Popularity Predictor - Project Summary

## 🎯 Results

**Model Performance:**
- ✅ **Test Accuracy: 63.14%** (baseline: 50%)
- Training Accuracy: 86.12%
- Model shows some overfitting but generalizes reasonably

**Classification Breakdown:**
- Unpopular posts: 66% precision, 77% recall
- Popular posts: 56% precision, 43% recall
- Model is better at identifying unpopular posts

## 📊 Dataset Analysis

**24,548 Reddit posts about "Clothing"**
- Text length: 2-16,176 characters (avg: 168)
- 1,168 unique subreddits
- Score range: -251 to 29,122 (median: 2.0)
- Popular threshold: score > 2 (40.6% of posts)

## 🧠 Model Architecture

**4-Layer Neural Network:**
```
Input (512 features)
  ↓
Hidden 1 (256 neurons + ReLU + BatchNorm + Dropout 0.3)
  ↓
Hidden 2 (128 neurons + ReLU + BatchNorm + Dropout 0.3)
  ↓
Hidden 3 (64 neurons + ReLU + Dropout 0.3)
  ↓
Output (1 neuron + Sigmoid)
```

**Features Used:**
- 12 numeric features (text stats, metadata)
- 500 TF-IDF text features from title + body
- Total: 512 input features

## 🔍 Key Learnings

**What Makes Posts Popular:**
1. **Text characteristics matter**
   - Length, word count, capitalization
   - Question marks and exclamation points
   
2. **Engagement signals**
   - Number of comments (logged)
   - Having body text vs title-only
   
3. **Content type**
   - Self posts vs links
   - Video posts
   - Flair presence

**Model Insights:**
- Model learns text patterns indicating popularity
- Overfitting suggests need for more data or regularization
- Binary classification is challenging - actual scores vary greatly

## 📁 Files Generated

```
side_project/
├── reddit_dataset.json          # Original 24K posts
├── train_popularity.py          # Training script
├── predict.py                   # Inference script
├── README.md                    # Project documentation
├── SUMMARY.md                   # This file
├── popularity_model.pt          # Trained model checkpoint
├── training_curves.png          # Loss & accuracy plots
├── feature_importance.png       # Top 20 features
└── confusion_matrix.png         # Performance visualization
```

## 💡 Improvements to Try

### Easy Wins:
1. **Add subreddit embeddings** - Categorical encoding of top subreddits
2. **Temporal features** - Day of week, hour posted, seasonality
3. **Author features** - Account age, karma (if available)
4. **Better text preprocessing** - Remove URLs, handle emojis

### Medium Effort:
5. **LSTM for text** - Better sequential understanding
6. **Ensemble models** - Combine multiple models
7. **Regression instead** - Predict actual score (harder but more useful)
8. **Class imbalance** - Use weighted loss or SMOTE

### Advanced:
9. **Pre-trained transformers** - Fine-tune BERT/DistilBERT
10. **Multi-task learning** - Predict score + sentiment + subreddit
11. **Active learning** - Label most uncertain predictions manually
12. **Cross-validation** - K-fold for better generalization estimate

## 🎓 Educational Value

**Skills Demonstrated:**
- ✅ Feature engineering from text and metadata
- ✅ TF-IDF vectorization for NLP
- ✅ PyTorch neural network implementation
- ✅ Data preprocessing and scaling
- ✅ Train/test split and evaluation
- ✅ Visualization of results
- ✅ Model persistence and loading
- ✅ Building inference pipeline

**Concepts Applied:**
- Binary classification
- Natural Language Processing (NLP)
- Deep learning with PyTorch
- Regularization (dropout, weight decay)
- Learning rate scheduling
- Feature importance analysis

## 🚀 Next Steps

1. **Try the predictor** - Run `python predict.py` with your own posts
2. **Experiment with hyperparameters** - Adjust hidden sizes, dropout rates
3. **Add more features** - Implement temporal or categorical features
4. **Try regression** - Predict actual scores instead of binary
5. **Deploy as API** - Create a Flask/FastAPI endpoint

---

**Created:** November 24, 2025  
**Model:** 4-layer MLP with 512 input features  
**Performance:** 63.14% test accuracy on 24.5K Reddit posts
