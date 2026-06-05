import requests, json, os, re
from dotenv import load_dotenv
load_dotenv()

GROQ_TOKEN = os.getenv("GROQ_TOKEN")
TAVILY_KEY = os.getenv("TAVILY_KEY")   # images only
EXA_KEY    = os.getenv("EXA_KEY")      # web search + social


# ─────────────────────────────────────────
#  GROQ
# ─────────────────────────────────────────
def ask_groq(prompt, max_tokens=3000):
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_TOKEN}"},
            json={"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens},
            timeout=30,
        )
        
        # Check if request was successful
        if r.status_code != 200:
            error_msg = f"API Error {r.status_code}: {r.text[:200]}"
            print(error_msg)
            return f'{{"action": "answer", "understood": "API error occurred", "search_query": "{prompt[:100]}", "image_search_query": "", "direct_url": "", "quantity": 5, "answer": "Sorry, I encountered an error: {error_msg}"}}'
        
        response_json = r.json()
        
        # Validate response structure
        if "choices" not in response_json:
            print(f"Unexpected API response: {response_json}")
            return f'{{"action": "answer", "understood": "Invalid API response", "search_query": "{prompt[:100]}", "image_search_query": "", "direct_url": "", "quantity": 5, "answer": "Sorry, I received an invalid response from the AI service."}}'
        
        if not response_json["choices"]:
            print("Empty choices array")
            return f'{{"action": "answer", "understood": "No response from API", "search_query": "{prompt[:100]}", "image_search_query": "", "direct_url": "", "quantity": 5, "answer": "Sorry, I didn't receive a valid response."}}'
        
        if "message" not in response_json["choices"][0]:
            print(f"Missing message in choice: {response_json['choices'][0]}")
            return f'{{"action": "answer", "understood": "Invalid message format", "search_query": "{prompt[:100]}", "image_search_query": "", "direct_url": "", "quantity": 5, "answer": "Sorry, the response format was invalid."}}'
        
        if "content" not in response_json["choices"][0]["message"]:
            print(f"Missing content in message: {response_json['choices'][0]['message']}")
            return f'{{"action": "answer", "understood": "No content in response", "search_query": "{prompt[:100]}", "image_search_query": "", "direct_url": "", "quantity": 5, "answer": "Sorry, the response contained no content."}}'
        
        return response_json["choices"][0]["message"]["content"].strip()
    
    except requests.exceptions.Timeout:
        print("Groq API timeout")
        return f'{{"action": "answer", "understood": "Request timeout", "search_query": "{prompt[:100]}", "image_search_query": "", "direct_url": "", "quantity": 5, "answer": "Sorry, the request timed out. Please try again."}}'
    
    except requests.exceptions.ConnectionError:
        print("Failed to connect to Groq API")
        return f'{{"action": "answer", "understood": "Connection failed", "search_query": "{prompt[:100]}", "image_search_query": "", "direct_url": "", "quantity": 5, "answer": "Sorry, I cannot connect to the AI service. Please check your internet connection."}}'
    
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}, response text: {r.text[:200] if 'r' in locals() else 'No response'}")
        return f'{{"action": "answer", "understood": "Invalid JSON response", "search_query": "{prompt[:100]}", "image_search_query": "", "direct_url": "", "quantity": 5, "answer": "Sorry, I received an invalid response format."}}'
    
    except Exception as e:
        print(f"Unexpected error in ask_groq: {str(e)}")
        return f'{{"action": "answer", "understood": "Unexpected error", "search_query": "{prompt[:100]}", "image_search_query": "", "direct_url": "", "quantity": 5, "answer": "Sorry, an unexpected error occurred: {str(e)}"}}'


# ─────────────────────────────────────────
#  QUANTITY EXTRACTOR
# ─────────────────────────────────────────
def extract_quantity(query: str):
    word_map = {"one":1,"two":2,"three":3,"four":4,"five":5,
                "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
                "eleven":11,"twelve":12,"fifteen":15,"twenty":20}
    q = query.lower()
    for word, num in word_map.items():
        if re.search(r'\b'+word+r'\b', q):
            return num
    m = re.search(r'\b(\d{1,2})\s*(?:image|photo|picture|pic|result|link|item|news|article)s?\b', q)
    if m: return int(m.group(1))
    m2 = re.search(r'\bgive\s+me\s+(\d{1,2})\b', q)
    if m2: return int(m2.group(1))
    return None


# ─────────────────────────────────────────
#  EXA — web search
# ─────────────────────────────────────────
def exa_search(query: str, num: int = 5, include_text: bool = True):
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
            headers={"x-api-key": EXA_KEY, "Content-Type": "application/json"},
            json=payload, timeout=30,
        )
        data = r.json()
        results = []
        for item in data.get("results", []):
            url = item.get("url", "")
            try: domain = url.split("/")[2].replace("www.", "")
            except: domain = ""
            highlights = item.get("highlights", [])
            desc = " ".join(highlights)[:350] if highlights else (item.get("text") or "")[:350]
            results.append({
                "title": item.get("title", ""),
                "url": url,
                "description": desc,
                "image": item.get("image", "") or "",
                "domain": domain,
            })
        return results
    except Exception as e:
        print(f"[exa error] {e}")
        return []


# ─────────────────────────────────────────
#  TAVILY — images only
# ─────────────────────────────────────────
def tavily_images(query: str, qty: int):
    try:
        tv = requests.post(
            "https://api.tavily.com/search",
            json={"api_key":TAVILY_KEY,"query":query,"search_depth":"advanced",
                  "include_images":True,"include_image_descriptions":True,"max_results":qty+10},
            timeout=30,
        ).json()
        src_map = {}
        for item in tv.get("results", []):
            try:
                dom = item["url"].split("/")[2].replace("www.", "")
                src_map[dom] = item["url"]
            except: pass
        images = []
        for img in tv.get("images", []):
            url = img if isinstance(img, str) else img.get("url", "")
            if not url.startswith("http"): continue
            dom = next((d for d in src_map if d.lower() in url.lower()), "")
            if not dom and src_map:
                keys = list(src_map.keys())
                dom = keys[min(len(images), len(keys)-1)]
            images.append({"url": url, "source": dom})
            if len(images) >= qty: break
        return images
    except Exception as e:
        print(f"[tavily image error] {e}")
        return []


# ─────────────────────────────────────────
#  SOCIAL MEDIA META — fetch OG tags
# ─────────────────────────────────────────
SOCIAL_DOMAINS = {
    "tiktok": "tiktok.com",
    "youtube": "youtube.com",
    "youtu.be": "youtube.com",
    "instagram": "instagram.com",
    "twitter": "twitter.com",
    "x.com": "x.com",
    "facebook": "facebook.com",
}

SOCIAL_COLORS = {
    "tiktok":    {"bg":"#010101","text":"#ffffff","label":"TikTok"},
    "youtube":   {"bg":"#ff0000","text":"#ffffff","label":"YouTube"},
    "instagram": {"bg":"#e1306c","text":"#ffffff","label":"Instagram"},
    "twitter":   {"bg":"#1da1f2","text":"#ffffff","label":"Twitter / X"},
    "x.com":     {"bg":"#000000","text":"#ffffff","label":"X"},
    "facebook":  {"bg":"#1877f2","text":"#ffffff","label":"Facebook"},
}

SOCIAL_URL_PATTERNS = [
    (r'(https?://)?(www\.)?tiktok\.com/@[\w.]+/video/\d+', 'tiktok'),
    (r'(https?://)?(vm\.)?tiktok\.com/\w+', 'tiktok'),
    (r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+', 'youtube'),
    (r'(https?://)?(www\.)?youtu\.be/[\w-]+', 'youtube'),
    (r'(https?://)?(www\.)?instagram\.com/(?:p|reel|tv)/[\w-]+', 'instagram'),
    (r'(https?://)?(www\.)?twitter\.com/\w+/status/\d+', 'twitter'),
    (r'(https?://)?(www\.)?x\.com/\w+/status/\d+', 'x.com'),
    (r'(https?://)?(www\.)?facebook\.com/(?:watch|reel|video)', 'facebook'),
]

def detect_social_url(query: str):
    """Returns (full_url, platform) or (None, None)."""
    for pat, platform in SOCIAL_URL_PATTERNS:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            raw = m.group(0)
            if not raw.startswith("http"):
                raw = "https://" + raw
            return raw, platform
    return None, None

def fetch_og_meta(url: str, platform: str) -> dict:
    """Fetch Open Graph / oEmbed metadata for a social URL."""
    meta = {"url": url, "platform": platform, "title": "", "description": "",
            "thumbnail": "", "author": "", "color": SOCIAL_COLORS.get(platform, {"bg":"#333","text":"#fff","label":platform.capitalize()})}
    try:
        # Try oEmbed for YouTube
        if platform == "youtube":
            vid_match = re.search(r'(?:v=|youtu\.be/)([\w-]+)', url)
            if vid_match:
                vid = vid_match.group(1)
                oe = requests.get(f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={vid}&format=json", timeout=10).json()
                meta["title"] = oe.get("title", "")
                meta["author"] = oe.get("author_name", "")
                meta["thumbnail"] = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
                return meta

        # Generic OG tag scrape
        headers = {"User-Agent": "Mozilla/5.0 (compatible; nmedea/1.0)"}
        resp = requests.get(url, headers=headers, timeout=12, allow_redirects=True)
        html = resp.text
        def og(prop):
            m = re.search(r'<meta[^>]+(?:property|name)=["\']og:'+prop+r'["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
            if not m:
                m = re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']og:'+prop+r'["\']', html, re.IGNORECASE)
            return m.group(1).strip() if m else ""
        meta["title"]       = og("title") or re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE) and re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE).group(1) or ""
        meta["description"] = og("description")
        meta["thumbnail"]   = og("image")
    except Exception as e:
        print(f"[og fetch error] {e}")
    return meta

def find_social_video(query: str, platform: str) -> dict | None:
    """Search EXA for a video on a specific platform."""
    clean = re.sub(r'\b(find|show|get|search|look up|bring|fetch|me|a|an|the|video|videos|on|from|in)\b', '', query, flags=re.IGNORECASE).strip()
    site = SOCIAL_DOMAINS.get(platform, platform+".com")
    results = exa_search(f'site:{site} {clean}', num=3, include_text=False)
    if not results:
        return None
    top = results[0]
    return {
        "url": top["url"],
        "platform": platform,
        "title": top["title"],
        "description": top["description"],
        "thumbnail": top.get("image", ""),
        "author": "",
        "color": SOCIAL_COLORS.get(platform, {"bg":"#333","text":"#fff","label":platform.capitalize()}),
    }


# ─────────────────────────────────────────
#  MAIN UNDERSTAND
# ─────────────────────────────────────────
def understand(query: str):
    explicit_qty = extract_quantity(query)

    # ── Social URL pasted directly? ──
    social_url, social_platform = detect_social_url(query)

    # ── Looking for a video on a platform? ──
    find_match = re.search(
        r'\b(find|show|get|search|look up|bring|fetch)\b.{0,40}\b(tiktok|youtube|instagram|twitter|x\.com|facebook)\b',
        query, re.IGNORECASE
    )
    find_platform = find_match.group(2).lower().replace(".", "") if find_match else None

    # If it's purely a social URL with no other text, handle directly
    if social_url and len(query.strip().split()) <= 6:
        social_meta = fetch_og_meta(social_url, social_platform)
        return {
            "action": "social",
            "url": social_url,
            "understood": f"Show {social_platform} content",
            "answer": "",
            "images": [],
            "results": [],
            "sources": [],
            "social": social_meta,
        }

    prompt = f"""You are the world's most powerful AI search engine. User said: "{query}"

Reply ONLY this JSON, nothing else:
{{
  "action": "answer OR open_website OR show_images OR show_results OR answer_with_images",
  "understood": "what user wants in one clear sentence",
  "search_query": "perfect specific search query matching EXACTLY what user asked",
  "image_search_query": "very specific image query — include gender/age/style/quality descriptors, never swap subject",
  "direct_url": "full URL if user wants to open a site, else empty string",
  "quantity": {explicit_qty if explicit_qty is not None else 5},
  "answer": "if action is answer or answer_with_images: COMPLETE detailed response, no word limit. else empty string"
}}

Rules:
- open/go to/visit site → action=open_website
- wants images/photos/pictures only → action=show_images, answer=""
- wants images AND information → action=answer_with_images
- asks a question or general query → action=answer with FULL answer
- wants list/results/links → action=show_results
- quantity = EXACTLY {explicit_qty if explicit_qty is not None else 5}
- image_search_query MUST be faithful to subject gender/type/style"""

    text = ask_groq(prompt)
    start, end = text.find('{'), text.rfind('}')+1
    data = json.loads(text[start:end])

    action = data.get("action", "answer")
    qty = max(1, min(explicit_qty or int(data.get("quantity", 5)), 20))

    if action == "open_website":
        return {
            "action": "open_website",
            "url": data.get("direct_url", ""),
            "understood": data.get("understood", ""),
            "answer": "", "images": [], "results": [], "sources": [], "social": None,
        }

    images, results, sources = [], [], []
    answer = data.get("answer", "")
    social_meta = None

    # ── Social video search ──
    if find_platform:
        social_meta = find_social_video(query, find_platform)

    # ── Social URL in longer query ──
    if social_url and not social_meta:
        social_meta = fetch_og_meta(social_url, social_platform)

    # ── IMAGES via Tavily ──
    if action in ["show_images", "answer_with_images"]:
        img_q = data.get("image_search_query") or data.get("search_query", query)
        images = tavily_images(img_q, qty)

    # ── WEB RESULTS via EXA ──
    exa_n = qty if action == "show_results" else 5
    exa_results = exa_search(data.get("search_query", query), num=exa_n)

    if action == "show_results":
        results = exa_results[:qty]
    elif action in ["answer", "answer_with_images"]:
        results = exa_results[:4]
    elif action == "show_images":
        results = exa_results[:4]

    for item in exa_results[:6]:
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
        "social": social_meta,
    }