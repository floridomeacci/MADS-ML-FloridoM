import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

# Load the model
print("Loading trained model...")
checkpoint = torch.load('popularity_model.pt', weights_only=False)
threshold = checkpoint['threshold']
scaler = checkpoint['scaler']
tfidf = checkpoint['tfidf']
feature_names = checkpoint['feature_names']

print(f"Model threshold for 'Very Popular': score > {threshold}")

# Load model architecture
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

def predict_post(title, body, num_comments=0):
    """Predict how popular a post will be"""
    
    # Create feature dictionary
    text = title + ' ' + body
    
    features = {
        'text_length': len(text),
        'title_length': len(title),
        'body_length': len(body),
        'word_count': len(text.split()),
        'has_body': 1 if len(body) > 0 else 0,
        'title_uppercase_ratio': sum(1 for c in title if c.isupper()) / max(len(title), 1),
        'has_question_mark': 1 if '?' in text else 0,
        'has_exclamation': 1 if '!' in text else 0,
        'num_comments_log': np.log1p(num_comments),
        'is_video': 0,
        'is_self': 1,
        'has_flair': 0
    }
    
    # TF-IDF features
    text_features = tfidf.transform([text]).toarray()
    
    # Combine features
    X_numeric = np.array([[features[f] for f in feature_names]])
    X = np.hstack([X_numeric, text_features])
    X_scaled = scaler.transform(X)
    
    # Predict
    with torch.no_grad():
        logit = model(torch.FloatTensor(X_scaled))
        probability = torch.sigmoid(logit).item()
    
    return {
        'probability': probability,
        'is_popular': probability > 0.5,
        'confidence': probability if probability > 0.5 else (1 - probability),
        'features': features
    }

# Test posts
test_posts = [
    {
        'title': 'I just found a paradise planet to settle my home base. Lets populate it!',
        'body': '''Just discovered an amazing paradise planet in Euclid! No storms, perfect weather, beautiful flora and fauna. 

Thinking of building a community hub here. Who wants to join?

Coordinates coming soon!''',
        'subreddit': 'NoMansSkyTheGame',
        'name': 'POST 1: Gaming Community'
    },
    {
        'title': 'How to make loops not loose..?',
        'body': '''I'm trying to repair my favorite sweater with a visible patch but the loops keep coming loose. 

Any tips for keeping the stitches tight? I'm using embroidery thread on knit fabric.

Thanks in advance!''',
        'subreddit': 'Visiblemending',
        'name': 'POST 2: Help Request'
    },
    {
        'title': 'Finally got the big plushie',
        'body': '''After months of searching, I finally found the giant version I wanted!

So happy with this purchase. Worth the wait!

Anyone else collecting the big ones?''',
        'subreddit': 'plushies',
        'name': 'POST 3: Collection Share'
    }
]

print("\n" + "="*80)
print("🧪 TESTING RECOMMENDED POSTS")
print("="*80)

results = []
for i, post in enumerate(test_posts, 1):
    result = predict_post(post['title'], post['body'])
    results.append((post, result))
    
    print(f"\n{'='*80}")
    print(f"📝 {post['name']}")
    print(f"{'='*80}")
    print(f"Title: {post['title']}")
    print(f"Subreddit: r/{post['subreddit']}")
    print(f"\nBody:")
    print(post['body'])
    print(f"\n{'─'*80}")
    print(f"🤖 MODEL PREDICTION:")
    print(f"{'─'*80}")
    
    if result['is_popular']:
        print(f"✅ WILL BE POPULAR (top 15%)")
        print(f"📊 Confidence: {result['confidence']*100:.1f}%")
    else:
        print(f"❌ LIKELY UNPOPULAR")
        print(f"📊 Confidence: {result['confidence']*100:.1f}%")
    
    print(f"📈 Probability: {result['probability']*100:.2f}%")
    print(f"📏 Text length: {result['features']['text_length']} chars")
    print(f"📝 Word count: {result['features']['word_count']} words")
    print(f"❓ Has question: {'Yes' if result['features']['has_question_mark'] else 'No'}")
    print(f"❗ Has exclamation: {'Yes' if result['features']['has_exclamation'] else 'No'}")

# Summary
print("\n" + "="*80)
print("📊 SUMMARY & RECOMMENDATION")
print("="*80)

# Sort by probability
results.sort(key=lambda x: x[1]['probability'], reverse=True)

print("\n🏆 Ranking by popularity probability:")
for i, (post, result) in enumerate(results, 1):
    status = "✅ POPULAR" if result['is_popular'] else "❌ UNPOPULAR"
    print(f"{i}. {post['name']:30s} | {result['probability']*100:6.2f}% | {status}")

best_post, best_result = results[0]
print(f"\n🎯 **BEST CHOICE TO TEST:**")
print(f"   Post: {best_post['name']}")
print(f"   Subreddit: r/{best_post['subreddit']}")
print(f"   Predicted probability: {best_result['probability']*100:.2f}%")

if best_result['is_popular']:
    print(f"\n   ✅ The model predicts this WILL be popular!")
else:
    print(f"\n   ⚠️  The model is conservative - even this is predicted unpopular")
    print(f"   But it's your BEST SHOT with {best_result['probability']*100:.2f}% chance")

print(f"\n📋 Copy and paste this to Reddit:")
print("="*80)
print(f"SUBREDDIT: r/{best_post['subreddit']}")
print(f"\nTITLE:\n{best_post['title']}")
print(f"\nBODY:\n{best_post['body']}")
print("="*80)
print("\n💡 After posting, come back and tell me the score to see if the model was right!")
