import joblib
import re
import sys

def clean_input(text):
    return re.sub(r'\W+', ' ', text).lower().strip()

def predict_message(message):
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/nb_model.pkl")

    cleaned = clean_input(message)
    vect_message = vectorizer.transform([cleaned])
    prediction = model.predict(vect_message)

    result = "SPAM" if prediction[0] == 1 else "HAM (not spam)"
    print(f"Prediction: {result}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, help="Enter the message to classify")
    args = parser.parse_args()

    if args.message:
        predict_message(args.message)
    else:
        print("Please provide a message using the --message argument.")
