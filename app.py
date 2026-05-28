from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
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
    prompt = f"""The user is typing in a search bar: "{query}"
Give 5 smart AI-powered search suggestions that complete or extend what they might be looking for.
Reply with only a JSON array of 5 strings, nothing else. Example: ["suggestion 1","suggestion 2","suggestion 3","suggestion 4","suggestion 5"]"""
    try:
        text = ask_groq(prompt)
        start = text.find('[')
        end = text.rfind(']') + 1
        suggestions = json.loads(text[start:end])
        return jsonify({"suggestions": suggestions[:5]})
    except:
        return jsonify({"suggestions": []})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
