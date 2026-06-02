import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_TOKEN = os.getenv("GROQ_TOKEN")
TAVILY_KEY = os.getenv("TAVILY_KEY")
MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return {"queries": []}

def save_memory(memory):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

def ask_groq(prompt):
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

def understand(query):
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
            domain = r.get("url","").split("/")[2].replace("www.","")
            sources.append({"domain": domain, "url": r.get("url","")})
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