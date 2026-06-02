from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_TOKEN = os.getenv("GROQ_TOKEN")
TAVILY_KEY = os.getenv("TAVILY_KEY")

app = Flask(__name__, static_folder='templates', static_url_path='')
CORS(app)

MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return {"queries": [], "vision_history": []}

def save_memory(memory):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

def ask_groq(prompt):
    """Call Groq API for text completion"""
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_TOKEN}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            },
            timeout=30
        )
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return ""

def ask_groq_vision(image_base64, prompt):
    """Use Groq's Llama 3.2 Vision model to describe an image"""
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    
    headers = {
        "Authorization": f"Bearer {GROQ_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.2-11b-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        result = response.json()
        if "error" in result:
            print(f"Groq Vision error: {result}")
            return "I'm having trouble seeing right now. Please try again."
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Vision error: {e}")
        return "I couldn't process this image. Please try again with better lighting."

def understand(query):
    """Process user query and return appropriate response"""
    memory = load_memory()
    memory["queries"].append(query)
    save_memory(memory)

    prompt = f"""You are an AI browser assistant. The user said: "{query}"

Analyze exactly what they want and respond in this exact JSON only:
{{
  "action": "one of: answer, open_website, show_images, show_results",
  "understood": "one sentence of what user wants",
  "search_query": "best search query for this",
  "direct_url": "if user said open/go to a specific site put full URL here else empty string",
  "quantity": 5,
  "answer": "if action is answer write a full natural conversational response here else empty string"
}}

Rules:
- If user says open/go to a website: action=open_website, put the URL in direct_url
- If user asks for images: action=show_images
- If user asks a question: action=answer, write the full answer in the answer field
- If user wants results/recommendations: action=show_results
- quantity means how many results or images they want, default 5
- Never add any text outside the JSON"""

    text = ask_groq(prompt)
    start = text.find('{')
    end = text.rfind('}') + 1
    data = json.loads(text[start:end])

    if data["action"] == "open_website":
        return {
            "action": "open_website",
            "url": data.get("direct_url", ""),
            "understood": data.get("understood", ""),
            "answer": "",
            "images": [],
            "results": [],
            "sources": []
        }

    if data["action"] == "answer":
        return {
            "action": "answer",
            "url": "",
            "understood": data.get("understood", ""),
            "answer": data.get("answer", ""),
            "images": [],
            "results": [],
            "sources": []
        }

    # Search with Tavily
    try:
        tavily = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_KEY,
                "query": data["search_query"],
                "search_depth": "advanced",
                "include_images": data["action"] == "show_images",
                "include_answer": True,
                "max_results": data.get("quantity", 5)
            },
            timeout=30
        ).json()
    except Exception as e:
        print(f"Tavily error: {e}")
        tavily = {"images": [], "results": [], "answer": ""}

    images = []
    if data["action"] == "show_images":
        for img in tavily.get("images", []):
            src = img if isinstance(img, str) else img.get("url", "")
            if src:
                images.append(src)
        images = images[:data.get("quantity", 5)]

    results = []
    if data["action"] == "show_results":
        for r in tavily.get("results", [])[:data.get("quantity", 5)]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("content", "")[:200]
            })

    sources = []
    for r in tavily.get("results", [])[:5]:
        try:
            domain = r.get("url", "").split("/")[2].replace("www.", "")
            sources.append({"domain": domain, "url": r.get("url", "")})
        except:
            pass

    answer = tavily.get("answer", "") or data.get("understood", "")

    return {
        "action": data["action"],
        "url": "",
        "understood": data.get("understood", ""),
        "answer": answer,
        "images": images,
        "results": results,
        "sources": sources
    }

@app.route("/")
def home():
    return send_from_directory('templates', 'index.html')

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    result = understand(query)
    return jsonify(result)

@app.route("/suggest", methods=["POST"])
def suggest():
    data = request.json
    query = data.get("query", "")
    if len(query) < 2:
        return jsonify({"suggestions": []})
    
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
        return jsonify({"suggestions": [
            f"{query} meaning",
            f"what is {query}",
            f"{query} explained",
            f"{query} definition",
            f"{query} examples"
        ][:5]})

@app.route("/vision", methods=["POST"])
def vision():
    """Handle camera image and return AI description"""
    data = request.json
    image_base64 = data.get('image')
    user_query = data.get('query', 'Describe what you see in detail')
    
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400
    
    memory = load_memory()
    if "vision_history" not in memory:
        memory["vision_history"] = []
    
    prompt = f"""You are nmedea, an AI assistant with vision capabilities. 
The user says: "{user_query}"

Describe what you see in this image in a natural, helpful way. 
Be specific about objects, people, colors, actions, and context.
Keep it conversational but informative. 
Respond as if you're talking directly to the user."""

    description = ask_groq_vision(image_base64, prompt)
    
    memory["vision_history"].insert(0, {
        "description": description,
        "timestamp": data.get('timestamp', '')
    })
    memory["vision_history"] = memory["vision_history"][:20]
    save_memory(memory)
    
    return jsonify({
        "description": description,
        "success": True
    })

@app.route("/history", methods=["GET"])
def get_history():
    memory = load_memory()
    return jsonify({
        "queries": memory.get("queries", [])[-10:],
        "vision_history": memory.get("vision_history", [])[:10]
    })

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)