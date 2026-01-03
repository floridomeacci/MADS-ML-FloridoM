import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import json

# Load the dataset
print("Loading Reddit dataset...")
with open('reddit_dataset.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Filter to text posts (same as training)
print("\nFiltering to text posts...")
initial_count = len(df)
df = df[df['is_video'] != True].copy()
df = df[df['body'].fillna('').str.len() >= 10].copy()
print(f"Text posts (no video, has body): {len(df)} (removed {initial_count - len(df)} media/link posts)")

# Engineer same features as training
print("\nEngineering features...")
df['text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')
df['text_length'] = df['text'].str.len()
df['title_length'] = df['title'].fillna('').str.len()
df['body_length'] = df['body'].fillna('').str.len()
df['word_count'] = df['text'].str.split().str.len()
df['has_body'] = (df['body'].fillna('').str.len() > 0).astype(int)
df['title_uppercase_ratio'] = df['title'].fillna('').apply(lambda x: sum(c.isupper() for c in x) / len(x) if len(x) > 0 else 0)
df['has_question_mark'] = df['title'].fillna('').str.contains('\?', regex=True).astype(int)
df['has_exclamation'] = df['title'].fillna('').str.contains('!', regex=True).astype(int)
df['num_comments_log'] = np.log1p(df['num_comments'])
df['is_video'] = df['is_video'].fillna(False).astype(int)
df['is_self'] = df['is_self'].fillna(False).astype(int)
df['has_flair'] = df.get('link_flair_text', pd.Series([None]*len(df))).notna().astype(int)

# Load the model
print("\nLoading trained model...")
checkpoint = torch.load('popularity_model.pt', weights_only=False)
threshold = checkpoint['threshold']
scaler = checkpoint['scaler']
tfidf = checkpoint['tfidf']
feature_names = checkpoint['feature_names']

print(f"Model predicts 'Very Popular' for posts with score > {threshold} (top 15%)")

# Define actual labels
df['is_popular'] = (df['score'] > threshold).astype(int)

# Prepare features
text_features = tfidf.transform(df['text'].fillna('')).toarray()
X_numeric = df[feature_names].values
X = np.hstack([X_numeric, text_features])
X_scaled = scaler.transform(X)

# Load model architecture
import torch.nn as nn

class PopularityPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=512, dropout=0.4):
        super(PopularityPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

model = PopularityPredictor(checkpoint['input_size'], hidden_size=512, dropout=0.4)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
print("\nMaking predictions on all posts...")
with torch.no_grad():
    logits = model(torch.FloatTensor(X_scaled))
    probabilities = torch.sigmoid(logits).numpy()
    predictions = (probabilities > 0.5).astype(int)

# Add predictions to dataframe
df['predicted_popular'] = predictions
df['probability'] = probabilities

# Calculate accuracy
accuracy = (df['is_popular'] == df['predicted_popular']).mean()
print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

# Show some interesting examples
print("\n" + "="*100)
print("SAMPLE PREDICTIONS vs ACTUAL RESULTS")
print("="*100)

# Show 5 correct predictions for popular posts
print("\n✅ CORRECTLY PREDICTED AS POPULAR (5 examples):")
print("-"*100)
correct_popular = df[(df['is_popular'] == 1) & (df['predicted_popular'] == 1)].sample(min(5, len(df[(df['is_popular'] == 1) & (df['predicted_popular'] == 1)])))
for idx, row in correct_popular.iterrows():
    title = str(row['title']) if pd.notna(row['title']) else "No title"
    print(f"\nTitle: {title[:80]}...")
    print(f"Actual Score: {row['score']} | Predicted: POPULAR ({row['probability']*100:.1f}% confidence) | ✅ CORRECT")
    print(f"Comments: {row['num_comments']} | Subreddit: r/{row['subreddit']}")

# Show 5 correct predictions for unpopular posts
print("\n" + "-"*100)
print("\n✅ CORRECTLY PREDICTED AS UNPOPULAR (5 examples):")
print("-"*100)
correct_unpopular = df[(df['is_popular'] == 0) & (df['predicted_popular'] == 0)].sample(min(5, len(df[(df['is_popular'] == 0) & (df['predicted_popular'] == 0)])))
for idx, row in correct_unpopular.iterrows():
    title = str(row['title']) if pd.notna(row['title']) else "No title"
    print(f"\nTitle: {title[:80]}...")
    print(f"Actual Score: {row['score']} | Predicted: UNPOPULAR ({(1-row['probability'])*100:.1f}% confidence) | ✅ CORRECT")
    print(f"Comments: {row['num_comments']} | Subreddit: r/{row['subreddit']}")

# Show false positives (predicted popular but actually unpopular)
print("\n" + "-"*100)
print("\n❌ FALSE POSITIVES - Predicted POPULAR but was UNPOPULAR (5 examples):")
print("-"*100)
false_positives = df[(df['is_popular'] == 0) & (df['predicted_popular'] == 1)]
if len(false_positives) > 0:
    for idx, row in false_positives.sample(min(5, len(false_positives))).iterrows():
        title = str(row['title']) if pd.notna(row['title']) else "No title"
        print(f"\nTitle: {title[:80]}...")
        print(f"Actual Score: {row['score']} | Predicted: POPULAR ({row['probability']*100:.1f}% confidence) | ❌ WRONG")
        print(f"Comments: {row['num_comments']} | Subreddit: r/{row['subreddit']}")
else:
    print("No false positives!")

# Show false negatives (predicted unpopular but actually popular)
print("\n" + "-"*100)
print("\n❌ FALSE NEGATIVES - Predicted UNPOPULAR but was POPULAR (5 examples):")
print("-"*100)
false_negatives = df[(df['is_popular'] == 1) & (df['predicted_popular'] == 0)]
if len(false_negatives) > 0:
    for idx, row in false_negatives.sample(min(5, len(false_negatives))).iterrows():
        title = str(row['title']) if pd.notna(row['title']) else "No title"
        print(f"\nTitle: {title[:80]}...")
        print(f"Actual Score: {row['score']} | Predicted: UNPOPULAR ({(1-row['probability'])*100:.1f}% confidence) | ❌ WRONG")
        print(f"Comments: {row['num_comments']} | Subreddit: r/{row['subreddit']}")
else:
    print("No false negatives!")

# Statistics
print("\n" + "="*100)
print("STATISTICS")
print("="*100)
print(f"Total posts: {len(df)}")
print(f"Actually popular (score > {threshold}): {df['is_popular'].sum()} ({df['is_popular'].mean()*100:.1f}%)")
print(f"Predicted popular: {df['predicted_popular'].sum()} ({df['predicted_popular'].mean()*100:.1f}%)")
print(f"\nCorrect predictions: {(df['is_popular'] == df['predicted_popular']).sum()}")
print(f"False positives: {len(false_positives)}")
print(f"False negatives: {len(false_negatives)}")
print(f"\nAccuracy: {accuracy*100:.2f}%")
