from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

# Download NLTK data on startup
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

from intent import understand, ask_groq
import json

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")
    result = understand(query)
    return jsonify(result)

@app.route("/suggest", methods=["POST"])
def suggest():
    data = request.json
    query = data.get("query", "")
    prompt = f"""User is typing: "{query}"
Give 5 smart search suggestions. Reply ONLY a JSON array of 5 strings, nothing else:
["suggestion 1","suggestion 2","suggestion 3","suggestion 4","suggestion 5"]"""
    try:
        text = ask_groq(prompt)
        start = text.find('[')
        end = text.rfind(']') + 1
        suggestions = json.loads(text[start:end])
        return jsonify({"suggestions": suggestions[:5]})
    except:
        return jsonify({"suggestions": []})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
