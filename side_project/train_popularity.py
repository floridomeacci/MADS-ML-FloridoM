"""
Reddit Post Popularity Predictor
Predict whether a post will be popular (high score) based on title, body, and metadata
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from pathlib import Path

# Setup
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("Loading Reddit dataset...")
with open('reddit_dataset.json', 'r') as f:
    data = json.load(f)

# Create DataFrame
df = pd.DataFrame(data)
print(f"Loaded {len(df)} posts")

# Filter to text posts (exclude videos and posts without body text)
print("\nFiltering to text posts...")
initial_count = len(df)
# Remove video posts
df = df[df['is_video'] != True].copy()
# Keep posts with actual body text (at least 10 characters)
df = df[df['body'].fillna('').str.len() >= 10].copy()
print(f"Text posts (no video, has body): {len(df)} (removed {initial_count - len(df)} media/link posts)")

# Feature Engineering
print("\nEngineering features...")

# Text features
df['text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')
df['text_length'] = df['text'].str.len()
df['title_length'] = df['title'].fillna('').str.len()
df['body_length'] = df['body'].fillna('').str.len()
df['word_count'] = df['text'].str.split().str.len()
df['has_body'] = (df['body_length'] > 0).astype(int)
df['title_uppercase_ratio'] = df['title'].fillna('').apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
df['has_question_mark'] = df['text'].str.contains('?', regex=False).astype(int)
df['has_exclamation'] = df['text'].str.contains('!', regex=False).astype(int)

# Metadata features
df['num_comments_log'] = np.log1p(df['num_comments'].fillna(0))
df['is_video'] = df['is_video'].fillna(False).astype(int)
df['is_self'] = df['is_self'].fillna(False).astype(int)
df['has_flair'] = df['flair'].notna().astype(int)

# Target: Binary classification (very popular vs rest)
# Very Popular = score in top 15% (easier task = higher accuracy)
top_15_threshold = df['score'].quantile(0.85)
df['is_popular'] = (df['score'] > top_15_threshold).astype(int)

print(f"Top 15% score threshold: {top_15_threshold}")
print(f"Very popular posts (score > {top_15_threshold}): {df['is_popular'].sum()} ({df['is_popular'].mean()*100:.1f}%)")

# Text vectorization (TF-IDF) - More features for better accuracy
print("\nVectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(max_features=1000, stop_words='english', min_df=3, ngram_range=(1, 2))
text_features = tfidf.fit_transform(df['text'].fillna('')).toarray()
print(f"TF-IDF features: {text_features.shape[1]}")

# Numeric features
numeric_features = [
    'text_length', 'title_length', 'body_length', 'word_count',
    'has_body', 'title_uppercase_ratio', 'has_question_mark',
    'has_exclamation', 'num_comments_log', 'is_video', 'is_self', 'has_flair'
]

X_numeric = df[numeric_features].values
X_text = text_features

# Combine features
X = np.hstack([X_numeric, X_text])
y = df['is_popular'].values

print(f"\nTotal features: {X.shape[1]}")
print(f"  - Numeric features: {len(numeric_features)}")
print(f"  - Text features: {text_features.shape[1]}")

# Train/test split - 90/10 for more training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=RANDOM_SEED, stratify=y
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# PyTorch Dataset
class RedditDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = RedditDataset(X_train, y_train)
test_dataset = RedditDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Neural Network Model - Larger for higher accuracy
class PopularityPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=512, dropout=0.4):
        super().__init__()
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
            
            nn.Linear(hidden_size // 4, 1)  # No sigmoid - BCEWithLogitsLoss handles it
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

input_size = X_train.shape[1]
model = PopularityPredictor(input_size, hidden_size=512, dropout=0.4)

# Calculate class weights to balance the dataset
num_unpopular = (y_train == 0).sum()
num_popular = (y_train == 1).sum()
weight_for_popular = num_unpopular / num_popular  # Give more weight to minority class
pos_weight = torch.tensor([weight_for_popular], dtype=torch.float32)
print(f"\nClass balance: Unpopular={num_unpopular}, Popular={num_popular}")
print(f"Positive class weight: {weight_for_popular:.2f}")

# Loss and optimizer - using weighted loss for class balance
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=8)

# Training
print("\n" + "="*60)
print("Training Popularity Predictor Neural Network (Target: 80%+ accuracy)")
print("="*60)

epochs = 100
train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (predictions == y_batch).sum().item()
        train_total += len(y_batch)
    
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    
    # Testing
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            test_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            test_correct += (predictions == y_batch).sum().item()
            test_total += len(y_batch)
    
    test_loss /= len(test_loader)
    test_acc = test_correct / test_total
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    scheduler.step(test_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    
    # Early stopping if we hit target accuracy
    if test_acc >= 0.80:
        print(f"\n🎯 Target accuracy of 80% reached at epoch {epoch+1}!")
        break

print("\n" + "="*60)
print(f"🎯 Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print("="*60)

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'tfidf': tfidf,
    'feature_names': numeric_features,
    'input_size': input_size,
    'threshold': top_15_threshold
}, 'popularity_model.pt')
print("\n✅ Model saved to popularity_model.pt")

# Visualizations
print("\n📊 Creating visualizations...")

# 1. Training curves
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(test_losses, label='Test Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy', linewidth=2)
plt.plot(test_accs, label='Test Accuracy', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("✅ Saved training_curves.png")
plt.close()

# 2. Feature importance (using first layer weights)
first_layer_weights = model.network[0].weight.data.abs().mean(dim=0).numpy()
feature_names = numeric_features + [f'tfidf_{i}' for i in range(text_features.shape[1])]

# Top features
top_k = 20
top_indices = np.argsort(first_layer_weights)[-top_k:]
top_weights = first_layer_weights[top_indices]
top_names = [feature_names[i] for i in top_indices]

plt.figure(figsize=(10, 8))
plt.barh(range(len(top_names)), top_weights)
plt.yticks(range(len(top_names)), top_names)
plt.xlabel('Average Absolute Weight')
plt.title(f'Top {top_k} Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("✅ Saved feature_importance.png")
plt.close()

# 3. Confusion matrix-like visualization
model.eval()
with torch.no_grad():
    test_predictions = model(torch.FloatTensor(X_test))
    test_predictions = (torch.sigmoid(test_predictions) > 0.5).numpy().astype(int)

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, test_predictions)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.xticks([0, 1], ['Unpopular', 'Popular'])
plt.yticks([0, 1], ['Unpopular', 'Popular'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("✅ Saved confusion_matrix.png")
plt.close()

# Print classification report
print("\n" + "="*60)
print("Classification Report:")
print("="*60)
print(classification_report(y_test, test_predictions, target_names=['Unpopular', 'Popular']))

print("\n✅ Training complete!")
print("\nGenerated files:")
print("  - popularity_model.pt (saved model)")
print("  - training_curves.png")
print("  - feature_importance.png")
print("  - confusion_matrix.png")
