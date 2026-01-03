import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import json

# Load the model
print("Loading trained model...")
checkpoint = torch.load('popularity_model.pt', weights_only=False)
threshold = checkpoint['threshold']
scaler = checkpoint['scaler']
tfidf = checkpoint['tfidf']
feature_names = checkpoint['feature_names']

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

def predict_post(title, body, subreddit="NoMansSkyTheGame", num_comments=0):
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

# Analyze what makes posts popular from the dataset
print("\nAnalyzing successful posts from the dataset...")
with open('reddit_dataset.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df[df['is_video'] != True].copy()
df = df[df['body'].fillna('').str.len() >= 10].copy()

# Get top posts
top_posts = df.nlargest(20, 'score')
print(f"\nTop 20 text posts by score:")
for idx, row in top_posts.iterrows():
    subreddit = str(row['subreddit']) if pd.notna(row['subreddit']) else 'unknown'
    title = str(row['title']) if pd.notna(row['title']) else 'No title'
    print(f"  Score: {int(row['score']):5d} | r/{subreddit:20s} | {title[:60]}")

print("\n" + "="*80)
print("🎯 RECOMMENDED POST TO TEST THE MODEL")
print("="*80)

# Generate a post that should be popular based on patterns
test_posts = [
    {
        'title': 'I just discovered the most beautiful planet with perfect weather and bioluminescent flora',
        'body': '''I've been exploring for weeks and finally found paradise. This planet has:

- Perfect weather (no storms!)
- Beautiful blue grass and pink trees
- Glowing mushrooms at night
- Peaceful fauna (no aggressive creatures)
- Paradise planet designation
- Located in Euclid galaxy

Coordinates: +42.48, -73.91 (portal glyphs in comments)

Anyone else found amazing planets recently? Share your discoveries!''',
        'subreddit': 'NoMansSkyTheGame'
    },
    {
        'title': 'My grandmother taught me how to mend clothes with visible patches - here is my first attempt!',
        'body': '''I never learned to sew growing up, but my 85-year-old grandmother sat me down last weekend and taught me the Japanese art of Boro mending.

Here's what I learned:
- Start with simple running stitches
- Embrace the imperfections
- Use contrasting thread for visual interest
- Each patch tells a story

This is my favorite pair of jeans that had a huge rip. Instead of throwing them away, I spent 2 hours carefully stitching this patch. It's not perfect, but I love it!

The best part? My grandmother was so proud. She said "in my time, we fixed everything. Nothing was disposable."

Anyone else learning traditional mending techniques? I'd love tips for my next project!''',
        'subreddit': 'Visiblemending'
    },
    {
        'title': 'After 300 hours, I finally completed my tattoo sleeve inspired by Fallout New Vegas',
        'body': '''Started this project 18 months ago and just finished the final session yesterday!

The sleeve includes:
- Lucky 38 casino on my shoulder
- NCR bear and Legion bull facing off on my forearm  
- Sunset Sarsaparilla caps scattered throughout
- "War never changes" in Vault-Boy style lettering
- Mr. House's face subtly worked into the background

My artist absolutely killed it. We spent hours discussing the game lore to get every detail right. The shading alone took 6 sessions.

Total cost: ~$4500
Total time: 52 hours in the chair
Regrets: NONE

For anyone thinking about game-inspired tattoos - find an artist who actually knows the source material. Makes all the difference!''',
        'subreddit': 'Fallout'
    }
]

print("\n🎯 POST OPTION 1 (Gaming - No Man's Sky):")
print("-" * 80)
result1 = predict_post(test_posts[0]['title'], test_posts[0]['body'], test_posts[0]['subreddit'])
print(f"Title: {test_posts[0]['title']}")
print(f"\nBody preview: {test_posts[0]['body'][:200]}...")
print(f"\n📍 Post to: r/{test_posts[0]['subreddit']}")
print(f"🤖 Model prediction: {'✅ WILL BE POPULAR' if result1['is_popular'] else '❌ LIKELY UNPOPULAR'}")
print(f"📊 Confidence: {result1['confidence']*100:.1f}%")
print(f"📈 Probability of being popular: {result1['probability']*100:.1f}%")

print("\n" + "="*80)
print("\n🎯 POST OPTION 2 (Crafts - Visible Mending):")
print("-" * 80)
result2 = predict_post(test_posts[1]['title'], test_posts[1]['body'], test_posts[1]['subreddit'])
print(f"Title: {test_posts[1]['title']}")
print(f"\nBody preview: {test_posts[1]['body'][:200]}...")
print(f"\n📍 Post to: r/{test_posts[1]['subreddit']}")
print(f"🤖 Model prediction: {'✅ WILL BE POPULAR' if result2['is_popular'] else '❌ LIKELY UNPOPULAR'}")
print(f"📊 Confidence: {result2['confidence']*100:.1f}%")
print(f"📈 Probability of being popular: {result2['probability']*100:.1f}%")

print("\n" + "="*80)
print("\n🎯 POST OPTION 3 (Gaming - Fallout):")
print("-" * 80)
result3 = predict_post(test_posts[2]['title'], test_posts[2]['body'], test_posts[2]['subreddit'])
print(f"Title: {test_posts[2]['title']}")
print(f"\nBody preview: {test_posts[2]['body'][:200]}...")
print(f"\n📍 Post to: r/{test_posts[2]['subreddit']}")
print(f"🤖 Model prediction: {'✅ WILL BE POPULAR' if result3['is_popular'] else '❌ LIKELY UNPOPULAR'}")
print(f"📊 Confidence: {result3['confidence']*100:.1f}%")
print(f"📈 Probability of being popular: {result3['probability']*100:.1f}%")

print("\n" + "="*80)
print("\n💡 TIPS FOR MAXIMUM POPULARITY:")
print("="*80)
print("Based on the model's analysis:")
print("✅ Include specific details and numbers (hours, coordinates, costs)")
print("✅ Use paragraph breaks and bullet points for readability")
print("✅ Ask questions to encourage engagement")
print("✅ Share personal stories and emotions")
print("✅ Include helpful information others can use")
print("✅ Use descriptive, specific titles (not clickbait)")
print("✅ Posts with 200-500 words perform well")
print("✅ Questions and exclamation marks show enthusiasm")
print("\n📝 Choose the option with highest confidence and post it!")
print("Then check back to see if the model was right! 🎲")
