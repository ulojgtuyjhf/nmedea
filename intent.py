import requests, json, os, re
from dotenv import load_dotenv
load_dotenv()

GROQ_TOKEN = os.getenv("GROQ_TOKEN")
TAVILY_KEY = os.getenv("TAVILY_KEY")   # images only
EXA_KEY    = os.getenv("EXA_KEY")      # web search + social


# ─────────────────────────────────────────────────────────────────
#  GROQ
# ─────────────────────────────────────────────────────────────────
def ask_groq(prompt, max_tokens=3000):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_TOKEN}"},
        json={"model":"llama-3.3-70b-versatile",
              "messages":[{"role":"user","content":prompt}],
              "max_tokens":max_tokens},
        timeout=30,
    )
    return r.json()["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────
#  QUANTITY EXTRACTOR  — reads "100 images", "give me 10 photos", etc.
# ─────────────────────────────────────────────────────────────────
def extract_quantity(query: str):
    word_map = {
        "one":1,"two":2,"three":3,"four":4,"five":5,
        "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
        "eleven":11,"twelve":12,"fifteen":15,"twenty":20,
        "thirty":30,"fifty":50,"hundred":100,"a hundred":100,
    }
    q = query.lower()
    for word, num in word_map.items():
        if re.search(r'\b' + re.escape(word) + r'\b', q):
            return num
    # digit before or after image/photo/result/etc
    m = re.search(
        r'\b(\d{1,3})\s*(?:image|photo|picture|pic|result|link|item|news|article|video)s?\b',
        q)
    if m: return int(m.group(1))
    # "give me 25" / "show me 50"
    m2 = re.search(r'\b(?:give|show|fetch|find|get)\s+me\s+(\d{1,3})\b', q)
    if m2: return int(m2.group(1))
    # bare digit at start: "100 images of cats"
    m3 = re.search(r'^(\d{1,3})\s+\w', q)
    if m3: return int(m3.group(1))
    return None


# ─────────────────────────────────────────────────────────────────
#  EXA  — neural web search
# ─────────────────────────────────────────────────────────────────
def exa_search(query: str, num: int = 5, include_text: bool = True, site: str = None):
    try:
        q = f"site:{site} {query}" if site else query
        payload = {
            "query": q,
            "numResults": min(num, 10),
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


# ─────────────────────────────────────────────────────────────────
#  TAVILY  — images (supports large batches via multiple calls)
# ─────────────────────────────────────────────────────────────────
def tavily_images(query: str, qty: int):
    """Fetch up to `qty` images. Makes multiple Tavily calls if qty > 20."""
    all_images = []
    seen_urls = set()
    batch = min(qty + 10, 30)  # Tavily max per call

    # Run up to 5 calls to reach high qty (e.g. 100)
    calls_needed = min(5, (qty + batch - 1) // max(batch - 10, 1))
    for call_i in range(calls_needed):
        if len(all_images) >= qty:
            break
        # vary query slightly on later calls to get different images
        q = query if call_i == 0 else f"{query} {'hd photos' if call_i==1 else 'high resolution' if call_i==2 else 'gallery' if call_i==3 else 'collection'}"
        try:
            tv = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": TAVILY_KEY, "query": q,
                      "search_depth": "advanced",
                      "include_images": True,
                      "include_image_descriptions": True,
                      "max_results": batch},
                timeout=30,
            ).json()
            src_map = {}
            for item in tv.get("results", []):
                try:
                    dom = item["url"].split("/")[2].replace("www.", "")
                    src_map[dom] = item["url"]
                except: pass
            for img in tv.get("images", []):
                url = img if isinstance(img, str) else img.get("url", "")
                if not url.startswith("http") or url in seen_urls:
                    continue
                seen_urls.add(url)
                dom = next((d for d in src_map if d.lower() in url.lower()), "")
                if not dom and src_map:
                    keys = list(src_map.keys())
                    dom = keys[min(len(all_images), len(keys)-1)]
                all_images.append({"url": url, "source": dom})
                if len(all_images) >= qty:
                    break
        except Exception as e:
            print(f"[tavily image error call {call_i}] {e}")
    return all_images[:qty]


# ─────────────────────────────────────────────────────────────────
#  SOCIAL METADATA
# ─────────────────────────────────────────────────────────────────
SOCIAL_COLORS = {
    "tiktok":    {"bg":"#010101","text":"#ffffff","label":"TikTok"},
    "youtube":   {"bg":"#ff0000","text":"#ffffff","label":"YouTube"},
    "instagram": {"bg":"#e1306c","text":"#ffffff","label":"Instagram"},
    "twitter":   {"bg":"#1da1f2","text":"#ffffff","label":"Twitter / X"},
    "x":         {"bg":"#000000","text":"#ffffff","label":"X"},
    "facebook":  {"bg":"#1877f2","text":"#ffffff","label":"Facebook"},
    "reddit":    {"bg":"#ff4500","text":"#ffffff","label":"Reddit"},
}
SOCIAL_SITES = {
    "tiktok":"tiktok.com","youtube":"youtube.com","youtu.be":"youtube.com",
    "instagram":"instagram.com","twitter":"twitter.com","x":"x.com",
    "facebook":"facebook.com","reddit":"reddit.com",
}
SOCIAL_URL_RE = [
    (r'(?:https?://)?(?:www\.)?tiktok\.com/@[\w.]+/video/\d+', 'tiktok'),
    (r'(?:https?://)?(?:vm\.)?tiktok\.com/\w+', 'tiktok'),
    (r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+', 'youtube'),
    (r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+', 'youtube'),
    (r'(?:https?://)?(?:www\.)?instagram\.com/(?:p|reel|tv)/[\w-]+', 'instagram'),
    (r'(?:https?://)?(?:www\.)?twitter\.com/\w+/status/\d+', 'twitter'),
    (r'(?:https?://)?(?:www\.)?x\.com/\w+/status/\d+', 'x'),
    (r'(?:https?://)?(?:www\.)?facebook\.com/(?:watch|reel|video)', 'facebook'),
    (r'(?:https?://)?(?:www\.)?reddit\.com/r/\w+/comments/\w+', 'reddit'),
]

def detect_social_url(query):
    for pat, plat in SOCIAL_URL_RE:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            raw = m.group(0)
            if not raw.startswith("http"):
                raw = "https://" + raw
            return raw, plat
    return None, None

def detect_social_find(query):
    """Detect 'find me a tiktok about X' or 'search youtube for X'."""
    m = re.search(
        r'\b(?:find|show|get|search|look\s+up|bring|fetch)\b.{0,50}\b(tiktok|youtube|instagram|twitter|x\.com|facebook|reddit)\b',
        query, re.IGNORECASE)
    if m:
        plat = m.group(1).lower().replace(".com","").replace(".","")
        return plat
    # also handle "tiktok of X" / "youtube video about X"
    m2 = re.search(r'\b(tiktok|youtube|instagram|twitter|reddit)\b.{0,10}\b(?:of|about|on|for|with)\b', query, re.IGNORECASE)
    if m2:
        return m2.group(1).lower()
    return None

def fetch_og_meta(url, platform):
    meta = {
        "url": url, "platform": platform, "title": "", "description": "",
        "thumbnail": "", "author": "",
        "color": SOCIAL_COLORS.get(platform, {"bg":"#333","text":"#fff","label":platform.capitalize()}),
    }
    try:
        if platform == "youtube":
            vid = re.search(r'(?:v=|youtu\.be/)([\w-]+)', url)
            if vid:
                v = vid.group(1)
                oe = requests.get(
                    f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={v}&format=json",
                    timeout=10).json()
                meta["title"] = oe.get("title","")
                meta["author"] = oe.get("author_name","")
                meta["thumbnail"] = f"https://img.youtube.com/vi/{v}/hqdefault.jpg"
                return meta
        headers = {"User-Agent":"Mozilla/5.0 (compatible; nmedea/1.0; +https://nmedea.app)"}
        resp = requests.get(url, headers=headers, timeout=14, allow_redirects=True)
        html = resp.text
        def og(prop):
            for pat in [
                rf'<meta[^>]+property=["\']og:{prop}["\'][^>]+content=["\']([^"\']+)["\']',
                rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:{prop}["\']',
                rf'<meta[^>]+name=["\']og:{prop}["\'][^>]+content=["\']([^"\']+)["\']',
            ]:
                m = re.search(pat, html, re.IGNORECASE)
                if m: return m.group(1).strip()
            return ""
        title_m = re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE)
        meta["title"] = og("title") or (title_m.group(1) if title_m else "")
        meta["description"] = og("description")
        meta["thumbnail"] = og("image")
    except Exception as e:
        print(f"[og error] {e}")
    return meta

def find_social_video(query, platform):
    site = SOCIAL_SITES.get(platform, platform+".com")
    # strip intent words to get the actual subject
    clean = re.sub(
        r'\b(find|show|get|search|look\s+up|bring|fetch|me|a|an|the|video|videos|post|posts|on|from|in|about|of|for|with)\b',
        ' ', query, flags=re.IGNORECASE)
    clean = re.sub(r'\b'+re.escape(platform)+r'\b', '', clean, flags=re.IGNORECASE).strip()
    results = exa_search(clean, num=4, include_text=False, site=site)
    if not results:
        results = exa_search(f"{clean} {site}", num=4, include_text=False)
    if not results:
        return None
    top = results[0]
    return {
        "url": top["url"],
        "platform": platform,
        "title": top["title"],
        "description": top["description"],
        "thumbnail": top.get("image",""),
        "author": "",
        "color": SOCIAL_COLORS.get(platform, {"bg":"#333","text":"#fff","label":platform.capitalize()}),
        "all_results": results,  # extra links to show
    }


# ─────────────────────────────────────────────────────────────────
#  MAIN  understand()
# ─────────────────────────────────────────────────────────────────
def understand(query: str):
    explicit_qty = extract_quantity(query)

    # ── Social URL pasted directly? ──
    social_url, social_platform = detect_social_url(query)

    # ── Asking to FIND a social video? ──
    find_platform = detect_social_find(query)

    # Pure social URL — handle immediately, skip Groq
    if social_url and len(query.split()) <= 8:
        social_meta = fetch_og_meta(social_url, social_platform)
        return {
            "action": "social",
            "url": social_url,
            "understood": f"Show {social_platform} content",
            "answer": "", "images": [], "results": [], "sources": [],
            "social": social_meta,
        }

    # ── Pre-detect: user clearly wants to open a website ──
    open_match = re.search(
        r'\b(?:open|go\s+to|visit|take\s+me\s+to|navigate\s+to|launch)\b.{0,40}(?:\.com|\.org|\.net|\.co|\.io|youtube|google|facebook|twitter|instagram|tiktok|reddit|netflix|amazon|whatsapp)',
        query, re.IGNORECASE
    )
    # ── Pre-detect: user clearly wants image(s) ──
    image_match = re.search(
        r'\b(?:image|photo|picture|pic|show\s+me|give\s+me|fetch|find)\b.{0,30}\b(?:image|photo|picture|pic)s?\b',
        query, re.IGNORECASE
    ) or re.search(
        r'\b(?:image|photo|picture|pic)s?\s+of\b',
        query, re.IGNORECASE
    ) or (explicit_qty and re.search(r'\b(?:image|photo|picture|pic)s?\b', query, re.IGNORECASE))



Reply ONLY this JSON, nothing else:
{{
  "action": "answer OR open_website OR show_images OR show_results OR answer_with_images",
  "understood": "what user wants in one clear sentence",
  "search_query": "perfect specific search query matching EXACTLY what user asked — do not change subject, gender, topic",
  "image_search_query": "very specific image query — EXACT subject, gender, age, style. If user says boys write boys. If user says red car write red car. Never swap the subject.",
  "direct_url": "full URL if user wants to open a specific website e.g. https://youtube.com or https://google.com, else empty string",
  "quantity": {explicit_qty if explicit_qty is not None else 5},
  "answer": "if action is answer or answer_with_images: COMPLETE detailed response. else empty string"
}}

STRICT Rules — follow exactly:
1. User says 'open', 'go to', 'visit', 'take me to' + any site/app name → action=open_website, put full URL in direct_url
2. User mentions 'image', 'photo', 'picture', 'pic', 'show me' + noun → action=show_images, answer=""
3. User wants image AND explanation → action=answer_with_images
4. User asks a question or wants information → action=answer with full answer
5. User wants 'results', 'links', 'websites', 'list' → action=show_results
6. quantity = EXACTLY {explicit_qty if explicit_qty is not None else 5} — DO NOT change this number
7. If user says 'one image' or '1 image' quantity MUST be 1
8. NEVER swap the search subject — if user says boys return boys not girls"""

    text = ask_groq(prompt)
    start, end = text.find('{'), text.rfind('}')+1
    data = json.loads(text[start:end])

    action = data.get("action","answer")
    # Override with pre-detected intent if Groq got it wrong
    if open_match and action != "open_website":
        action = "open_website"
    if image_match and action not in ["show_images","answer_with_images"]:
        action = "show_images"
    qty = max(1, min(explicit_qty or int(data.get("quantity",5)), 100))

    if action == "open_website":
        url = data.get("direct_url","")
        if not url:
            # Try to extract a URL or site name from the query
            url_m = re.search(r'https?://\S+', query)
            if url_m:
                url = url_m.group(0)
            else:
                # Extract site name and build URL
                site_m = re.search(
                    r'\b(youtube|google|facebook|twitter|instagram|tiktok|reddit|netflix|amazon|whatsapp|github|linkedin|pinterest)\b',
                    query, re.IGNORECASE)
                if site_m:
                    url = 'https://www.' + site_m.group(1).lower() + '.com'
                else:
                    # Last resort: find domain-like word
                    dom_m = re.search(r'\b([\w-]+\.(?:com|org|net|co|io|app))\b', query, re.IGNORECASE)
                    if dom_m:
                        url = 'https://' + dom_m.group(1)
        return {
            "action":"open_website","url":url,
            "understood":data.get("understood",""),
            "answer":"","images":[],"results":[],"sources":[],"social":None,
        }

    images, results, sources = [], [], []
    answer = data.get("answer","")
    social_meta = None

    # ── Social find ──
    if find_platform:
        social_meta = find_social_video(query, find_platform)
        if social_meta:
            results = social_meta.pop("all_results", [])

    # ── Social URL in longer query ──
    if social_url and not social_meta:
        social_meta = fetch_og_meta(social_url, social_platform)

    # ── IMAGES via Tavily ──
    if action in ["show_images","answer_with_images"]:
        img_q = data.get("image_search_query") or data.get("search_query", query)
        images = tavily_images(img_q, qty)

    # ── WEB via EXA ──
    if not results:
        exa_n = qty if action == "show_results" else 5
        exa_results = exa_search(data.get("search_query", query), num=min(exa_n,10))
        results = exa_results[:qty] if action=="show_results" else exa_results[:4]

    exa_for_sources = exa_search(data.get("search_query",query), num=5) if not results else results
    for item in exa_for_sources[:6]:
        dom = item.get("domain","")
        if dom and not any(s["domain"]==dom for s in sources):
            sources.append({"domain":dom,"url":item["url"]})

    return {
        "action": action,
        "url": "",
        "understood": data.get("understood",""),
        "answer": answer,
        "images": images,
        "results": results,
        "sources": sources,
        "social": social_meta,
    }
