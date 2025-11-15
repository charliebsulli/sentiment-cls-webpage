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
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        if len(text) > 5000:
            return jsonify({"error": "Text exceeds maximum length of 5000 characters"}), 400
        
        if len(text) < 3:
            return jsonify({"error": "Text is too short, minimum length is 3 characters"}), 400
        
        features = vectorizer.transform([text])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = max(probability) * 100
        return jsonify({
            "sentiment": sentiment,
            "confidence": f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500