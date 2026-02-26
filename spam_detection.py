# Email Spam Detection Project

import pandas as pd
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['message'] = data['message'].apply(clean_text)

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Custom testing
def predict_email(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"

print("\n--- Test Your Own Email ---")
while True:
    message = input("\nEnter email message (or type exit): ")
    if message.lower() == "exit":
        break
    print("Prediction:", predict_email(message))