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

Reply ONLY this JSON:
{{
  "action": "answer OR open_website OR show_images OR show_results OR answer_with_images",
  "understood": "what user wants in one clear sentence",
  "search_query": "perfect search query for this",
  "image_search_query": "if images needed: specific image search query e.g. 'fast cars photos 4k'",
  "direct_url": "full URL if user wants to open a site, else empty",
  "quantity": 5,
  "answer": "if action is answer or answer_with_images: COMPLETE DETAILED response, no word limit, else empty"
}}

Rules:
- open/go to site -> action=open_website
- wants images/photos -> action=show_images, NO answer
- asks question AND wants images -> action=answer_with_images
- asks question/information -> action=answer
- wants list/results -> action=show_results
- quantity = EXACT number user requested, default 5"""

    text = ask_groq(prompt)
    start = text.find('{')
    end = text.rfind('}') + 1
    data = json.loads(text[start:end])

    action = data.get("action", "answer")
    qty = int(data.get("quantity", 5))

    if action == "open_website":
        return {"action":"open_website","url":data.get("direct_url",""),"understood":data.get("understood",""),"answer":"","images":[],"results":[],"sources":[]}

    images = []
    results = []
    sources = []
    answer = data.get("answer", "")

    needs_images = action in ["show_images", "answer_with_images"]
    needs_results = action == "show_results"
    needs_web = action in ["answer", "answer_with_images", "show_results"]

    if needs_images:
        # Fetch real images directly from web
        img_query = data.get("image_search_query") or data.get("search_query", query) + " photos"
        try:
            tv = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_KEY,
                    "query": img_query,
                    "search_depth": "advanced",
                    "include_images": True,
                    "include_image_descriptions": True,
                    "max_results": max(qty + 5, 10)
                },
                timeout=30
            ).json()

            for img in tv.get("images", []):
                if isinstance(img, str) and img.startswith("http"):
                    images.append({"url": img, "source": ""})
                elif isinstance(img, dict):
                    url = img.get("url", "")
                    if url.startswith("http"):
                        images.append({"url": url, "source": ""})

            # Match sources to images
            for i, r in enumerate(tv.get("results", [])):
                try:
                    domain = r.get("url","").split("/")[2].replace("www.","")
                    if i < len(images):
                        images[i]["source"] = domain
                    src_url = r.get("url","")
                    if not any(s["url"] == src_url for s in sources):
                        sources.append({"domain": domain, "url": src_url})
                except:
                    pass

            images = images[:qty]

        except Exception as e:
            pass

    if needs_web:
        try:
            tv = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_KEY,
                    "query": data.get("search_query", query),
                    "search_depth": "advanced",
                    "include_answer": True,
                    "max_results": qty if needs_results else 3
                },
                timeout=30
            ).json()

            if needs_results:
                for r in tv.get("results", [])[:qty]:
                    results.append({
                        "title": r.get("title",""),
                        "url": r.get("url",""),
                        "description": r.get("content","")[:300],
                        "image": r.get("image","")
                    })

            for r in tv.get("results", [])[:5]:
                try:
                    domain = r.get("url","").split("/")[2].replace("www.","")
                    if not any(s["domain"] == domain for s in sources):
                        sources.append({"domain": domain, "url": r.get("url","")})
                except:
                    pass

        except:
            pass

    return {
        "action": action,
        "url": "",
        "understood": data.get("understood",""),
        "answer": answer,
        "images": images,
        "results": results,
        "sources": sources
    }
