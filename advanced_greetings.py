import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset.
data = pd.read_csv("greetings.csv")

# Split the data into features and labels.
X = data["text"]
y = data["label"]

# Create a CountVectorizer object.
vectorizer = CountVectorizer()

# Transform the features.
X_vectorized = vectorizer.fit_transform(X)

# Create a Naive Bayes classifier.
classifier = MultinomialNB()

# Train the classifier.
classifier.fit(X_vectorized, y)

# Test the classifier.
predictions = classifier.predict(X_vectorized)

# Evaluate the classifier.
accuracy = np.mean(predictions == y)

print("Accuracy:", accuracy)
