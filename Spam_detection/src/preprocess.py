import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

import joblib


# Load the dataset
data_path = "data/spam.csv" 
df = pd.read_csv(data_path, encoding="latin-1")

# Keep only the necessary columns
df = df[['v1', 'v2']]

# Rename columns to more understandable names
df.columns = ['label', 'message']

# Display the first few rows to verify the data
print("Dataset loaded successfully!")
print(df.head())

# Check for any missing values
print("\nMissing values in dataset:\n", df.isnull().sum())

# Convert labels to binary values: 'spam' -> 1, 'ham' -> 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})
print("Converted labels to numeric values.\n")



# Function to clean message text
def clean_text(message):
    # Remove punctuation and numbers
    message = re.sub(r'[^a-zA-Z\s]', '', message)
    # Convert to lowercase
    message = message.lower()
    return message

# Apply the cleaning function to the 'message' column
df['message'] = df['message'].apply(clean_text)

print("Cleaned the message text.\n")
print(df.head())

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=3000)  # You can adjust the number of features

# Fit and transform the message data
X = vectorizer.fit_transform(df['message']).toarray()

# The target variable (labels)
y = df['label'].values

print("Completed TF-IDF vectorization. Feature matrix shape:", X.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
nb_classifier = MultinomialNB()

# Train the model
nb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Show classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model to a file
model_filename = "src/spam_classifier_model.pkl"
joblib.dump(nb_classifier, model_filename)

print(f"Model saved to {model_filename}")
