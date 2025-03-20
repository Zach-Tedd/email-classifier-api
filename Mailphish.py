#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import glob
import pandas as pd
import numpy as np
import re
import nltk


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# In[17]:


# Download stopwords for text processing
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords


# In[18]:


import os

spam_dir = 'spam/'  # Update with actual path
ham_dir = 'easy_ham/'  # Update with actual path

print("Spam directory:", os.path.abspath(spam_dir))
print("Ham directory:", os.path.abspath(ham_dir))

print("Spam files found:", len(os.listdir(spam_dir)))
print("Ham files found:", len(os.listdir(ham_dir)))


# In[19]:


def load_emails_from_directory(directory, label):
    """Loads all email files from a given directory and assigns a label."""
    emails = []
    for file_name in os.listdir(directory):  # Load all files, regardless of extension
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):  # Ensure it's a file
            with open(file_path, 'r', encoding='latin-1', errors='ignore') as file:
                emails.append(file.read())
    return pd.DataFrame({'text': emails, 'label': label})

spam_emails = load_emails_from_directory(spam_dir, label=1)
ham_emails = load_emails_from_directory(ham_dir, label=0)

df = pd.concat([spam_emails, ham_emails], ignore_index=True)
print(f"Total emails loaded: {len(df)}")


# In[20]:


import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Function to clean and preprocess email text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply preprocessing
df['text'] = df['text'].apply(preprocess_text)


# In[21]:


# Count phishing (1) and non-phishing (0) emails
spam_df = df[df['label'] == 1]  # Get phishing emails
df_balanced = pd.concat([df, spam_df], ignore_index=True)  # Duplicate phishing emails

# Shuffle dataset to mix original and duplicated emails
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the new distribution
print(df_balanced['label'].value_counts())


# In[22]:


# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(df_balanced['text'])
y = df_balanced['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#X = vectorizer.fit_transform(df['text'])
#y = df['label']  # Labels (1 = spam, 0 = non-spam)

# Split into training and testing sets (80% train, 20% test)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Feature transformation complete! Training samples:", X_train.shape[0])


# In[23]:


# Train the Naïve Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

#from sklearn.linear_model import LogisticRegression

# Train Logistic Regression model
#model = LogisticRegression(max_iter=500, class_weight='balanced')  # Auto-adjusts weights
#model.fit(X_train, y_train)

#model = LogisticRegression(max_iter=500)
#model.fit(X_train, y_train)

# Make predictions
#y_pred = model.predict(X_test)

# Evaluate performance
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Logistic Regression Accuracy: {accuracy:.4f}")


# In[24]:


def classify_email(email_text):
    email_text = preprocess_text(email_text)
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)[0]
    confidence = model.predict_proba(email_vector)[0][prediction]  # Get confidence score
    return f"{'Phishing/Spam 🚨' if prediction == 1 else 'Legitimate ✅'} (Confidence: {confidence:.2f})"

new_email = """Congratulations! You've won a $1000 gift card. Click here to claim now."""
print(classify_email(new_email))


# In[25]:


def classify_email(email_text, threshold=0.4):  # Lower threshold to 40%
    email_text = preprocess_text(email_text)
    email_vector = vectorizer.transform([email_text])
    proba = model.predict_proba(email_vector)[0][1]  # Probability of phishing
    prediction = 1 if proba >= threshold else 0
    return f"{'Phishing 🚨' if prediction == 1 else 'Legitimate ✅'} (Confidence: {proba:.2f})"

print(classify_email(new_email))


# In[26]:


import joblib

# Save model and vectorizer
joblib.dump(model, "email_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


# In[27]:


get_ipython().system('pip install flask flask-cors')


# In[ ]:


from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# ✅ Load the trained model and vectorizer
model = joblib.load("email_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/classify", methods=["POST"])
def classify_email():
    try:
        data = request.json  # Get JSON data
        email_text = data.get("text", "")

        if not email_text:
            return jsonify({"error": "No email text provided"}), 400

        # Transform the email text using the vectorizer
        email_features = vectorizer.transform([email_text])

        # Predict using the model
        prediction = model.predict(email_features)[0]
        confidence = model.predict_proba(email_features).max()

        # Convert prediction to label
        label = "Phishing 🚨" if prediction == 1 else "Legitimate ✅"

        return jsonify({"label": label, "confidence": round(float(confidence), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":
    # Disable the auto-restart issue in Jupyter
    import os
    os.environ["FLASK_RUN_FROM_CLI"] = "false"

    app.run(host="0.0.0.0", port=5000, debug=False)  # Turn off debug mode



# In[ ]:





# In[ ]:




