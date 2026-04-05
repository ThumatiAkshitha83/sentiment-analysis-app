import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# Load your text file
with open('TextAnalytics.txt', 'r') as f:
    data = f.readlines()

# Convert to dataframe
df = pd.DataFrame(data, columns=['text'])

# CLEANING FUNCTION
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

df['cleaned'] = df['text'].apply(clean_text)

# VADER for labels
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.5:
        return 'positive'
    elif score > 0:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['cleaned'].apply(get_sentiment)

# TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['cleaned'])

y = df['sentiment']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# SAVE FILES
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))

print("Model and vectorizer saved successfully!")