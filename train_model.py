import pandas as pd
import string
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("sms.tsv", sep='\t', header=None, names=['label', 'message'])
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['message'] = data['message'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")