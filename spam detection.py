# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 2: Load dataset
df = pd.read_csv("spam_sms.csv", encoding='latin-1')

# Step 3: Data cleaning
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Split data
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Convert text → numbers (TF-IDF)
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Step 6: Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Predict on test data
y_pred = model.predict(X_test)

# Step 8: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Step 9: User input prediction
while True:
    user_msg = input("\nEnter a message (or type 'exit' to quit): ")
    
    if user_msg.lower() == 'exit':
        print("Exiting...")
        break

    user_vector = vectorizer.transform([user_msg])
    result = model.predict(user_vector)

    if result[0] == 1:
        print("Result: SPAM ")
    else:
        print("Result: NOT SPAM ")