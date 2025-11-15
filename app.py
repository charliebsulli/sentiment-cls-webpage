import pickle
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    text = data.get("text", "")

    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = max(probability) * 100
    return jsonify({
        "sentiment": sentiment,
        "confidence": f"{confidence:.2f}%"
    })