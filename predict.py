import pickle
import re
import string

# Load the trained model
with open("models/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open("models/vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


def clean_text(text):
    """
    Clean the input text before prediction.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_news(news_text):
    """
    Predict whether the given news is Real or Fake.
    """
    cleaned_text = clean_text(news_text)

    text_vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(text_vector)[0]

    if prediction == 0:
        return "Fake News"
    else:
        return "Real News"


# Test the model (optional)
if __name__ == "__main__":
    sample_news = input("Enter a news article:\n\n")

    result = predict_news(sample_news)

    print("\nPrediction:", result)