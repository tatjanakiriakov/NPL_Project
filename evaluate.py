import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(data_path):
    df = pd.read_csv(data_path)
    X = df['Message']
    y = df['Label']

    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/nb_model.pkl")

    X_vec = vectorizer.transform(X)
    _, X_test, _, y_test = train_test_split(X_vec, y, test_size=0.2, stratify=y, random_state=42)

    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

if __name__ == "__main__":
    evaluate_model("data/Cleaned_SMSSpamCollection.csv")
