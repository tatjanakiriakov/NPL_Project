import pandas as pd
import re
import os

def clean_text(text):
    text = re.sub(r'\W+', ' ', text)
    text = text.lower().strip()
    return text

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path, sep='\t', header=None, names=['Label', 'Message'], encoding='latin-1')
    df.drop_duplicates(inplace=True)
    df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})
    df['Message'] = df['Message'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    input_path = "data/SMSSpamCollection.csv"
    output_path = "data/Cleaned_SMSSpamCollection.csv"
    os.makedirs("data", exist_ok=True)
    preprocess_data(input_path, output_path)
