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

    prompt = f"""You are an AI browser. User said: "{query}"

Reply ONLY this JSON, nothing else:
{{
  "action": "answer OR open_website OR show_images OR show_results",
  "understood": "one sentence what user wants",
  "search_query": "best search query",
  "direct_url": "full URL if user wants to open a site, else empty",
  "quantity": 5,
  "answer": "if action is answer, full response here, else empty"
}}

Rules:
- open/go to site -> action=open_website, direct_url=full URL
- wants images/photos/pictures -> action=show_images
- asks a question -> action=answer
- wants results/recommendations/list -> action=show_results
- quantity = exact number user asked for, default 5"""

    text = ask_groq(prompt)
    start = text.find('{')
    end = text.rfind('}') + 1
    data = json.loads(text[start:end])

    if data["action"] == "open_website":
        return {
            "action": "open_website",
            "url": data.get("direct_url", ""),
            "understood": data.get("understood", ""),
            "answer": "", "images": [], "results": [], "sources": []
        }

    if data["action"] == "answer":
        return {
            "action": "answer",
            "url": "",
            "understood": data.get("understood", ""),
            "answer": data.get("answer", ""),
            "images": [], "results": [], "sources": []
        }

    want_images = data["action"] == "show_images"
    qty = int(data.get("quantity", 5))

    tavily = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_KEY,
            "query": data["search_query"],
            "search_depth": "advanced",
            "include_images": True,
            "include_image_descriptions": True,
            "include_answer": True,
            "max_results": qty
        },
        timeout=30
    ).json()

    # extract images properly
    images = []
    for img in tavily.get("images", []):
        if isinstance(img, str) and img.startswith("http"):
            images.append(img)
        elif isinstance(img, dict):
            url = img.get("url", "")
            if url.startswith("http"):
                images.append(url)

    # also pull images from results
    for r in tavily.get("results", []):
        img = r.get("image", "")
        if img and img.startswith("http") and img not in images:
            images.append(img)

    images = images[:qty]

    results = []
    if not want_images:
        for r in tavily.get("results", [])[:qty]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("content", "")[:200],
                "image": r.get("image", "")
            })

    sources = []
    for r in tavily.get("results", [])[:5]:
        try:
            domain = r.get("url","").split("/")[2].replace("www.","")
            sources.append({"domain": domain, "url": r.get("url","")})
        except:
            pass

    return {
        "action": data["action"],
        "url": "",
        "understood": data.get("understood", ""),
        "answer": tavily.get("answer", "") or data.get("understood", ""),
        "images": images,
        "results": results,
        "sources": sources
    }
