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

def ask_groq(prompt, max_tokens=2000):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_TOKEN}"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        },
        timeout=30
    )
    return r.json()["choices"][0]["message"]["content"].strip()

def understand(query):
    memory = load_memory()
    memory["queries"].append(query)
    save_memory(memory)

    # Step 1: AI understands exactly what user wants
    intent_prompt = f"""You are the world's most powerful AI search engine. A user said: "{query}"

Analyze exactly what they want. Reply ONLY this JSON, nothing else:
{{
  "action": "answer OR open_website OR show_images OR show_results",
  "understood": "what the user wants in one clear sentence",
  "search_query": "the perfect search query to find exactly this",
  "direct_url": "if user wants to open a specific website, the full URL, else empty string",
  "quantity": 5,
  "language": "language to respond in if user specified, else english",
  "answer": "if action is answer: write a COMPLETE, DETAILED, UNLIMITED response covering everything about this topic. Be thorough, informative and natural. No word limits. else empty string"
}}

Rules:
- open/go to/visit a site -> action=open_website, direct_url=full URL
- wants images/photos/pictures -> action=show_images, quantity=exact number they asked for
- wants a list/results/recommendations -> action=show_results, quantity=exact number they asked for  
- asks any question or wants information -> action=answer, write COMPLETE unlimited answer
- quantity: use EXACTLY what user requested, default 5
- If user asks in another language, respond in that language
- NEVER truncate the answer. Write everything."""

    text = ask_groq(intent_prompt, max_tokens=3000)
    start = text.find('{')
    end = text.rfind('}') + 1
    data = json.loads(text[start:end])

    action = data.get("action", "answer")
    quantity = int(data.get("quantity", 5))

    if action == "open_website":
        return {
            "action": "open_website",
            "url": data.get("direct_url", ""),
            "understood": data.get("understood", ""),
            "answer": "",
            "images": [],
            "results": [],
            "sources": []
        }

    if action == "answer":
        # For pure answers, still search for context to make answer richer
        try:
            tavily = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_KEY,
                    "query": data["search_query"],
                    "search_depth": "advanced",
                    "include_answer": True,
                    "max_results": 3
                },
                timeout=20
            ).json()
            
            # Combine AI answer with web context for richer response
            web_context = tavily.get("answer", "")
            ai_answer = data.get("answer", "")
            
            if web_context and web_context not in ai_answer:
                full_answer = ai_answer
            else:
                full_answer = ai_answer

            sources = []
            for r in tavily.get("results", [])[:3]:
                try:
                    domain = r.get("url","").split("/")[2].replace("www.","")
                    sources.append({"domain": domain, "url": r.get("url","")})
                except:
                    pass

            return {
                "action": "answer",
                "url": "",
                "understood": data.get("understood", ""),
                "answer": full_answer,
                "images": [],
                "results": [],
                "sources": sources
            }
        except:
            return {
                "action": "answer",
                "url": "",
                "understood": data.get("understood", ""),
                "answer": data.get("answer", ""),
                "images": [],
                "results": [],
                "sources": []
            }

    # For images and results, use Tavily
    want_images = action == "show_images"

    tavily = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_KEY,
            "query": data["search_query"],
            "search_depth": "advanced",
            "include_images": True,
            "include_image_descriptions": True,
            "include_answer": True,
            "max_results": max(quantity, 7)
        },
        timeout=30
    ).json()

    # Extract images - get EXACTLY what was requested
    images = []
    for img in tavily.get("images", []):
        if isinstance(img, str) and img.startswith("http"):
            images.append(img)
        elif isinstance(img, dict):
            url = img.get("url", "")
            if url.startswith("http"):
                images.append(url)

    # Also get images from results
    for r in tavily.get("results", []):
        img = r.get("image", "")
        if img and img.startswith("http") and img not in images:
            images.append(img)

    images = images[:quantity]

    # Extract results
    results = []
    for r in tavily.get("results", [])[:quantity]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "description": r.get("content", "")[:300],
            "image": r.get("image", "")
        })

    # Sources
    sources = []
    for r in tavily.get("results", [])[:5]:
        try:
            domain = r.get("url","").split("/")[2].replace("www.","")
            sources.append({"domain": domain, "url": r.get("url","")})
        except:
            pass

    answer = tavily.get("answer", "") or data.get("understood", "")

    return {
        "action": action,
        "url": "",
        "understood": data.get("understood", ""),
        "answer": answer,
        "images": images,
        "results": results,
        "sources": sources
    }
