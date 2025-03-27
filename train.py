import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df['Message']
    y = df['Label']

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, stratify=y, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/nb_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("Model and vectorizer saved in 'models/' folder.")

if __name__ == "__main__":
    train_model("data/Cleaned_SMSSpamCollection.csv")
