import requests, json, os, re
from dotenv import load_dotenv
load_dotenv()

GROQ_TOKEN = os.getenv("GROQ_TOKEN")
TAVILY_KEY = os.getenv("TAVILY_KEY")   # kept for images only
EXA_KEY    = os.getenv("EXA_KEY")      # bc1b9f49-fa7f-477a-84ce-817576206011


# ─────────────────────────────────────────
#  GROQ
# ─────────────────────────────────────────
def ask_groq(prompt, max_tokens=3000):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_TOKEN}"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
        timeout=30,
    )
    return r.json()["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────
#  QUANTITY EXTRACTOR
# ─────────────────────────────────────────
def extract_quantity(query: str):
    word_map = {
        "one":1,"two":2,"three":3,"four":4,"five":5,
        "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
        "eleven":11,"twelve":12,"fifteen":15,"twenty":20,
    }
    q = query.lower()
    for word, num in word_map.items():
        if re.search(r'\b' + word + r'\b', q):
            return num
    m = re.search(
        r'\b(\d{1,2})\s*(?:image|photo|picture|pic|result|link|item|news|article)s?\b', q)
    if m:
        return int(m.group(1))
    m2 = re.search(r'\bgive\s+me\s+(\d{1,2})\b', q)
    if m2:
        return int(m2.group(1))
    return None


# ─────────────────────────────────────────
#  EXA  — web search + neural search
# ─────────────────────────────────────────
def exa_search(query: str, num: int = 5, include_text: bool = True):
    """
    Returns list of {title, url, description, image, domain}
    Uses EXA neural search for best semantic results.
    """
    try:
        payload = {
            "query": query,
            "numResults": num,
            "type": "neural",
            "useAutoprompt": True,
            "contents": {
                "text": {"maxCharacters": 400} if include_text else False,
                "highlights": {"numSentences": 2, "highlightsPerUrl": 1} if include_text else False,
            },
        }
        r = requests.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": EXA_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        data = r.json()
        results = []
        for item in data.get("results", []):
            url = item.get("url", "")
            try:
                domain = url.split("/")[2].replace("www.", "")
            except Exception:
                domain = ""
            # best description: highlights first, then text snippet, then summary
            desc = ""
            highlights = item.get("highlights", [])
            if highlights:
                desc = " ".join(highlights)[:350]
            if not desc:
                txt = item.get("text", "") or ""
                desc = txt[:350]
            results.append({
                "title":       item.get("title", ""),
                "url":         url,
                "description": desc,
                "image":       item.get("image", "") or "",
                "domain":      domain,
            })
        return results
    except Exception as e:
        print(f"[exa error] {e}")
        return []


# ─────────────────────────────────────────
#  TAVILY  — images only
# ─────────────────────────────────────────
def tavily_images(query: str, qty: int):
    try:
        tv = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_KEY,
                "query": query,
                "search_depth": "advanced",
                "include_images": True,
                "include_image_descriptions": True,
                "max_results": qty + 10,
            },
            timeout=30,
        ).json()

        # build domain map from web results
        src_map = {}
        for item in tv.get("results", []):
            try:
                dom = item["url"].split("/")[2].replace("www.", "")
                src_map[dom] = item["url"]
            except Exception:
                pass

        images = []
        for img in tv.get("images", []):
            url = img if isinstance(img, str) else img.get("url", "")
            if not url.startswith("http"):
                continue
            dom = next((d for d in src_map if d.lower() in url.lower()), "")
            if not dom and src_map:
                keys = list(src_map.keys())
                dom = keys[min(len(images), len(keys) - 1)]
            images.append({"url": url, "source": dom})
            if len(images) >= qty:
                break
        return images
    except Exception as e:
        print(f"[tavily image error] {e}")
        return []


# ─────────────────────────────────────────
#  MAIN UNDERSTAND
# ─────────────────────────────────────────
def understand(query: str):
    explicit_qty = extract_quantity(query)

    prompt = f"""You are the world's most powerful AI search engine. User said: "{query}"

Reply ONLY this JSON, nothing else:
{{
  "action": "answer OR open_website OR show_images OR show_results OR answer_with_images",
  "understood": "what user wants in one clear sentence",
  "search_query": "perfect specific search query matching EXACTLY what user asked",
  "image_search_query": "very specific image query — include gender/age/style/quality e.g. 'male teen basketball players 4k', never swap gender/subject",
  "direct_url": "full URL if user wants to open a site, else empty string",
  "quantity": {explicit_qty if explicit_qty is not None else 5},
  "answer": "if action is answer or answer_with_images: COMPLETE detailed response, no word limit. else empty string"
}}

Rules:
- open/go to/visit → action=open_website, direct_url=full URL
- wants images/photos/pictures only → action=show_images, answer=""
- wants images AND information → action=answer_with_images
- asks a question or general query → action=answer with FULL answer
- wants list/results/links → action=show_results
- quantity = EXACTLY {explicit_qty if explicit_qty is not None else 5}
- image_search_query MUST be extremely specific and faithful to subject"""

    text = ask_groq(prompt)
    start, end = text.find('{'), text.rfind('}') + 1
    data = json.loads(text[start:end])

    action = data.get("action", "answer")
    qty = max(1, min(explicit_qty or int(data.get("quantity", 5)), 20))

    if action == "open_website":
        return {
            "action": "open_website",
            "url": data.get("direct_url", ""),
            "understood": data.get("understood", ""),
            "answer": "", "images": [], "results": [], "sources": [],
        }

    images, results, sources = [], [], []
    answer = data.get("answer", "")

    # ── IMAGES via Tavily ──
    if action in ["show_images", "answer_with_images"]:
        img_q = data.get("image_search_query") or data.get("search_query", query)
        images = tavily_images(img_q, qty)

    # ── WEB RESULTS via EXA ──
    # Always fetch results for context/sources, regardless of action
    exa_n = qty if action == "show_results" else 5
    exa_results = exa_search(data.get("search_query", query), num=exa_n)

    if action == "show_results":
        results = exa_results[:qty]
    elif action in ["answer", "answer_with_images"]:
        # Always attach at least 4 supporting links alongside the answer
        results = exa_results[:4]

    # build sources from exa results
    for item in exa_results[:6]:
        dom = item.get("domain", "")
        if dom and not any(s["domain"] == dom for s in sources):
            sources.append({"domain": dom, "url": item["url"]})

    # If action was show_images with no answer, still attach web results as context
    if action == "show_images" and not results:
        ctx = exa_search(data.get("search_query", query), num=4)
        results = ctx[:4]
        for item in ctx:
            dom = item.get("domain", "")
            if dom and not any(s["domain"] == dom for s in sources):
                sources.append({"domain": dom, "url": item["url"]})

    return {
        "action": action,
        "url": "",
        "understood": data.get("understood", ""),
        "answer": answer,
        "images": images,
        "results": results,
        "sources": sources,
    }