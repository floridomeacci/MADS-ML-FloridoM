"""
Make predictions on new Reddit posts using trained popularity model
"""
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model
print("Loading trained model...")
checkpoint = torch.load('popularity_model.pt', weights_only=False)
scaler = checkpoint['scaler']
tfidf = checkpoint['tfidf']
feature_names = checkpoint['feature_names']
input_size = checkpoint['input_size']
median_score = checkpoint['median_score']

# Recreate model architecture
import torch.nn as nn

class PopularityPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
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
            
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

model = PopularityPredictor(input_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded! (Trained on posts with median score: {median_score})")

def predict_popularity(title, body="", num_comments=0, is_video=False, is_self=True, has_flair=False):
    """
    Predict if a post will be popular
    
    Args:
        title (str): Post title
        body (str): Post body text
        num_comments (int): Number of comments (use 0 for new posts)
        is_video (bool): Is this a video post?
        is_self (bool): Is this a self/text post?
        has_flair (bool): Does post have flair?
    
    Returns:
        dict: Prediction results with probability and classification
    """
    # Text features
    text = title + ' ' + body
    text_length = len(text)
    title_length = len(title)
    body_length = len(body)
    word_count = len(text.split())
    has_body = 1 if body_length > 0 else 0
    title_uppercase_ratio = sum(1 for c in title if c.isupper()) / max(len(title), 1)
    has_question_mark = 1 if '?' in text else 0
    has_exclamation = 1 if '!' in text else 0
    
    # Metadata features
    num_comments_log = np.log1p(num_comments)
    is_video_int = 1 if is_video else 0
    is_self_int = 1 if is_self else 0
    has_flair_int = 1 if has_flair else 0
    
    # Numeric features
    numeric_features = np.array([[
        text_length, title_length, body_length, word_count,
        has_body, title_uppercase_ratio, has_question_mark,
        has_exclamation, num_comments_log, is_video_int, is_self_int, has_flair_int
    ]])
    
    # Text features (TF-IDF)
    text_features = tfidf.transform([text]).toarray()
    
    # Combine
    X = np.hstack([numeric_features, text_features])
    X = scaler.transform(X)
    
    # Predict
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        probability = model(X_tensor).item()
    
    is_popular = probability > 0.5
    
    return {
        'is_popular': is_popular,
        'probability': probability,
        'confidence': abs(probability - 0.5) * 2,  # 0 to 1 scale
        'prediction': 'POPULAR' if is_popular else 'UNPOPULAR'
    }

# Example predictions
print("\n" + "="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)

examples = [
    {
        'title': 'Where can I find affordable winter coats?',
        'body': 'Looking for recommendations on stores with good winter coat sales',
    },
    {
        'title': 'HELP! Ruined my favorite shirt in the wash!!!',
        'body': 'My white shirt turned pink after washing with red socks. Any way to fix this?',
    },
    {
        'title': 'My clothing brand just launched - check it out!',
        'body': '',
    },
    {
        'title': 'What do you think of this outfit?',
        'body': 'Wearing a blue jacket with black jeans and white sneakers. Too basic?',
    }
]

for i, post in enumerate(examples, 1):
    print(f"\n--- Example {i} ---")
    print(f"Title: {post['title']}")
    if post['body']:
        print(f"Body: {post['body'][:80]}...")
    
    result = predict_popularity(post['title'], post['body'])
    
    print(f"\n🎯 Prediction: {result['prediction']}")
    print(f"   Probability: {result['probability']:.2%}")
    print(f"   Confidence: {result['confidence']:.2%}")

print("\n" + "="*60)
print("\n💡 Try your own prediction:")
print("   result = predict_popularity('Your title here', 'Your body text here')")
print("   print(result)")
