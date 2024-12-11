import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Rename columns for clarity
data.rename(columns={'Category': 'class', 'Message': 'message'}, inplace=True)

# Preprocess the text
def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing
data['cleaned_message'] = data['message'].apply(preprocess_text)

# Encode the labels (spam = 1, ham = 0)
data['label_encoded'] = data['class'].map({'spam': 1, 'ham': 0})

# Split dataset into features and target
X = data['cleaned_message']
y = data['label_encoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Save the vectorizer for later use
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

# Save preprocessed data for later use
X_train_vectorized.toarray().dump('X_train.npy')
X_test_vectorized.toarray().dump('X_test.npy')
y_train.to_numpy().dump('y_train.npy')
y_test.to_numpy().dump('y_test.npy')

print("Preprocessing completed. Data and vectorizer saved.")
