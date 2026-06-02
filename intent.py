import requests
import json
import os
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

GROQ_TOKEN = os.getenv("GROQ_TOKEN")
TAVILY_KEY = os.getenv("TAVILY_KEY")

app = Flask(__name__, static_folder='.')
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

def ask_groq_vision(image_base64, prompt):
    """Use Groq's Llama 3.2 Vision model to describe an image"""
    # Remove data:image/jpeg;base64, prefix if present
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
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Vision error: {e}")
        return "I couldn't process this image. Please try again with better lighting."

def search_tavily(query, max_results=5):
    """Search the web using Tavily"""
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_KEY,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "max_results": max_results
            },
            timeout=30
        )
        return response.json()
    except Exception as e:
        print(f"Tavily error: {e}")
        return {"answer": "Search unavailable", "results": []}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/vision', methods=['POST'])
def vision():
    """Handle camera image and return AI description with speech"""
    data = request.json
    image_base64 = data.get('image')
    user_query = data.get('query', 'Describe what you see in detail')
    
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400
    
    # Store in memory
    memory = load_memory()
    if "vision_history" not in memory:
        memory["vision_history"] = []
    
    # Get AI description using Groq Vision
    prompt = f"""You are nmedea, an AI assistant with vision capabilities. 
The user says: "{user_query}"

Describe what you see in this image in a natural, helpful way. 
Be specific about objects, people, colors, actions, and context.
Keep it conversational but informative. 
If you recognize anything interesting, mention it.
Respond as if you're talking directly to the user."""

    description = ask_groq_vision(image_base64, prompt)
    
    # Save to history
    memory["vision_history"].insert(0, {
        "description": description,
        "timestamp": data.get('timestamp', '')
    })
    memory["vision_history"] = memory["vision_history"][:20]  # keep last 20
    save_memory(memory)
    
    return jsonify({
        "description": description,
        "success": True
    })

@app.route('/search', methods=['POST'])
def search():
    """Regular text search using Tavily"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "No query"}), 400
    
    memory = load_memory()
    if "queries" not in memory:
        memory["queries"] = []
    memory["queries"].append(query)
    save_memory(memory)
    
    # First, understand intent with Groq
    intent_prompt = f"""User query: "{query}"
    
Analyze and respond in JSON only:
{{
    "action": "answer or search",
    "search_query": "optimized search query if action is search",
    "direct_answer": "if simple question, answer directly here"
}}"""

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_TOKEN}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": intent_prompt}],
                "max_tokens": 200
            },
            timeout=30
        )
        intent = json.loads(r.json()["choices"][0]["message"]["content"])
        
        if intent.get("action") == "answer" and intent.get("direct_answer"):
            return jsonify({
                "action": "answer",
                "answer": intent["direct_answer"],
                "sources": []
            })
        
        # Search with Tavily
        search_query = intent.get("search_query", query)
        tavily_result = search_tavily(search_query, 5)
        
        return jsonify({
            "action": "search",
            "answer": tavily_result.get("answer", ""),
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "description": r.get("content", "")[:300]
                }
                for r in tavily_result.get("results", [])[:5]
            ],
            "sources": [
                {"domain": r.get("url", "").split("/")[2].replace("www.", ""), "url": r.get("url", "")}
                for r in tavily_result.get("results", [])[:3]
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e), "answer": "Sorry, I couldn't process that."}), 500

@app.route('/history', methods=['GET'])
def get_history():
    memory = load_memory()
    return jsonify({
        "queries": memory.get("queries", [])[-10:],
        "vision_history": memory.get("vision_history", [])[:10]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)