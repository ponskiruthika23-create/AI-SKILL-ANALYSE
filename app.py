from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load AI model
classifier = pipeline("sentiment-analysis")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    answer = data.get("answer", "")

    result = classifier(answer)[0]

    if result['label'] == 'POSITIVE':
        score = 50
    else:
        score = 20

    return jsonify({
        "ai_score": score,
        "confidence": result['score']
    })

if __name__ == "__main__":
    app.run(debug=True)
