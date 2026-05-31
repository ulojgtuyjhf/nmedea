import requests, json, os
from dotenv import load_dotenv
load_dotenv()

GROQ_TOKEN = os.getenv("GROQ_TOKEN")
TAVILY_KEY = os.getenv("TAVILY_KEY")

def ask_groq(prompt, max_tokens=3000):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_TOKEN}"},
        json={"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens},
        timeout=30
    )
    return r.json()["choices"][0]["message"]["content"].strip()

def understand(query):
    prompt = f"""You are the world's most powerful AI search engine. User said: "{query}"

Reply ONLY this JSON, nothing else:
{{
  "action": "answer OR open_website OR show_images OR show_results OR answer_with_images",
  "understood": "what user wants in one clear sentence",
  "search_query": "perfect search query",
  "image_search_query": "specific image search query if images needed e.g. 'fast cars 4k photos'",
  "direct_url": "full URL if user wants to open a site, else empty string",
  "quantity": 5,
  "answer": "if action is answer or answer_with_images: write COMPLETE DETAILED response with no word limit, covering everything thoroughly. else empty string"
}}

Rules:
- open/go to/visit site -> action=open_website, direct_url=full URL
- wants images/photos/pictures -> action=show_images, NO answer field
- wants images AND answer together -> action=answer_with_images
- asks a question -> action=answer, FULL unlimited answer
- wants list/results/recommendations -> action=show_results
- quantity = EXACT number user requested, default 5"""

    text = ask_groq(prompt)
    start = text.find('{')
    end = text.rfind('}') + 1
    data = json.loads(text[start:end])

    action = data.get("action", "answer")
    qty = int(data.get("quantity", 5))

    if action == "open_website":
        return {"action":"open_website","url":data.get("direct_url",""),"understood":data.get("understood",""),"answer":"","images":[],"results":[],"sources":[]}

    images, results, sources = [], [], []
    answer = data.get("answer", "")

    # Fetch images directly from web
    if action in ["show_images", "answer_with_images"]:
        img_q = data.get("image_search_query") or data.get("search_query", query) + " photos high quality"
        try:
            tv = requests.post("https://api.tavily.com/search", json={
                "api_key": TAVILY_KEY,
                "query": img_q,
                "search_depth": "advanced",
                "include_images": True,
                "include_image_descriptions": True,
                "max_results": qty + 6
            }, timeout=30).json()

            src_map = {}
            for r in tv.get("results", []):
                try:
                    dom = r.get("url","").split("/")[2].replace("www.","")
                    src_map[dom] = r.get("url","")
                    if not any(s["domain"]==dom for s in sources):
                        sources.append({"domain":dom,"url":r.get("url","")})
                except: pass

            for img in tv.get("images", []):
                url = img if isinstance(img,str) else img.get("url","")
                if not url.startswith("http"): continue
                dom = ""
                for d in src_map:
                    if d.lower() in url.lower():
                        dom = d; break
                if not dom and sources:
                    dom = sources[min(len(images), len(sources)-1)]["domain"]
                images.append({"url": url, "source": dom})
                if len(images) >= qty: break

        except Exception as e:
            pass

    # Fetch results or web context
    if action in ["show_results", "answer", "answer_with_images"]:
        try:
            tv = requests.post("https://api.tavily.com/search", json={
                "api_key": TAVILY_KEY,
                "query": data.get("search_query", query),
                "search_depth": "advanced",
                "include_answer": True,
                "max_results": qty if action=="show_results" else 3
            }, timeout=30).json()

            if action == "show_results":
                for r in tv.get("results", [])[:qty]:
                    results.append({
                        "title": r.get("title",""),
                        "url": r.get("url",""),
                        "description": r.get("content","")[:300],
                        "image": r.get("image","")
                    })

            for r in tv.get("results", [])[:5]:
                try:
                    dom = r.get("url","").split("/")[2].replace("www.","")
                    if not any(s["domain"]==dom for s in sources):
                        sources.append({"domain":dom,"url":r.get("url","")})
                except: pass

        except: pass

    return {
        "action": action,
        "url": "",
        "understood": data.get("understood",""),
        "answer": answer,
        "images": images,
        "results": results,
        "sources": sources
    }
