from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

app = Flask(__name__)

# Load your already trained model & vectorizer
model = pickle.load(open("model/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]
    cleaned = clean_text(review)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    result = "Positive 😊" if prediction == 1 else "Negative 😡"
    return render_template("index.html", prediction=result, review=review)

if __name__ == "__main__":
    app.run(debug=True)
