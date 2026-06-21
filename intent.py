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
        json={"model":"openai/gpt-oss-120b",
              "messages":[{"role":"user","content":prompt}],
              "max_tokens":max_tokens},
        timeout=30,
    )
    data = r.json()
    if "choices" not in data:
        print(f"[groq error] status={r.status_code} body={data}")
        raise RuntimeError(f"Groq API error: {data.get('error', data)}")
    return data["choices"][0]["message"]["content"].strip()


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
#  WIKIPEDIA  — enrichment only, never replaces the main answer
# ─────────────────────────────────────────────────────────────────
def wikipedia_summary(topic: str):
    """Fetch a short factual summary + thumbnail for a likely entity/topic.
    Returns None if no matching page exists — caller should treat this as
    optional enrichment, not a required result."""
    try:
        title = requests.utils.quote(topic.strip().replace(" ", "_"))
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            headers={"User-Agent": "nmedea/1.0 (https://nmedea.onrender.com)"},
            timeout=8,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("type") == "disambiguation":
            return None
        extract = data.get("extract", "")
        if not extract or len(extract) < 40:
            return None
        return {
            "title": data.get("title", topic),
            "summary": extract,
            "thumbnail": (data.get("thumbnail") or {}).get("source", ""),
            "url": (data.get("content_urls", {}).get("desktop", {}) or {}).get("page", ""),
        }
    except Exception as e:
        print(f"[wikipedia error] {e}")
        return None


# ─────────────────────────────────────────────────────────────────
#  MAIN  understand()
# ─────────────────────────────────────────────────────────────────
def understand(query: str, force_action: str = None):
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
            "social": social_meta, "wiki": None, "code": None,
        }

    prompt = f"""You are the world's most powerful AI search engine. User said: "{query}"

Reply ONLY this JSON, nothing else:
{{
  "action": "answer OR open_website OR go_to_thing OR build_code OR show_images OR show_results OR answer_with_images",
  "understood": "what user wants in one clear sentence",
  "search_query": "perfect specific search query matching EXACTLY what user asked — do not change subject, gender, topic",
  "image_search_query": "very specific image query — preserve EXACT subject, gender, age, style — e.g. if user says boys, write boys NOT girls",
  "direct_url": "full URL ONLY if user named a literal known site by name/domain (e.g. youtube.com, amazon), else empty string",
  "thing_query": "if action=go_to_thing: identify the exact real-world thing being described, then build a search query for the SPECIFIC destination the user actually wants — read their exact words for the clue. If they say 'watch'/'stream' → find where to watch it. If they say 'trailer' → find the trailer. If they just say 'open'/'show me'/'take me to' with no extra clue → find that thing's main official/primary page (e.g. its official site, its Wikipedia, its IMDB, its store page — whatever is the single best canonical destination for that exact kind of thing). NEVER default to 'trailer' unless the user's words actually suggest video/trailer/watch. e.g. 'that movie about the moving train with the heist' → 'Bullet Train 2022 movie official page'. e.g. 'watch that movie about the heist on a train' → 'Bullet Train 2022 watch online streaming'. e.g. 'that song that goes na na na hey hey' → 'Hey Baby na na na song'. e.g. 'the new iPhone' → 'iPhone 17 official Apple page'. Be as specific and literal to their actual request as possible.",
  "thing_type": "if action=go_to_thing: one of movie, show, song, product, app, game, person, place, other",
  "code_title": "if action=build_code: a short 3-6 word title for what the code does, e.g. 'Python To-Do List App'",
  "code_language": "if action=build_code: the language as a lowercase string, e.g. python, javascript, html, css, java, cpp",
  "code_content": "if action=build_code: the COMPLETE, working, runnable code — no placeholders, no '...rest of code', no TODOs. Fully finished.",
  "code_explainer": "if action=build_code: a short 2-3 sentence explanation of what the code does and how to use/run it",
  "quantity": {explicit_qty if explicit_qty is not None else 5},
  "answer": "if action is answer or answer_with_images: a clear, concise answer — 2-4 short sentences for simple facts, max 2 short paragraphs for anything more complex. Get straight to the point, no filler intro, no restating the question. else empty string"
}}

Decide the action using these grounded examples — match the closest pattern, don't overthink it:

action=open_website
  "open youtube.com" / "go to wikipedia" / "take me to amazon"
  → ONLY when the user names a specific, known site/domain literally. direct_url = that real domain.

action=go_to_thing
  "open that movie about the moving train" / "show me that song that goes na na na hey hey" / "take me to the new iPhone page"
  "find that show with the dragons" / "open the trailer for the new spiderman movie" / "go to that game everyone's playing, the purple one"
  → ANY time the user describes something (movie, show, song, product, app, game, person, place) they want to be taken DIRECTLY to, rather than asking a question about it or wanting a list of links about it
  → trigger words: open/show me/take me to/go to/find + a vague/descriptive reference to a specific real thing, NOT a literal URL
  → this is the user wanting to land INSIDE the actual thing (its trailer, its store page, its official page) — not read about it, not get a list

action=build_code
  "write me a python script that..." / "code a to-do list app" / "build a snake game in javascript" / "give me HTML for a landing page" / "create a function that sorts an array"
  → ANY request to write, build, code, or generate a program, script, function, app, website, or anything clearly asking for SOURCE CODE as the output
  → code_content must be COMPLETE and working — never truncated, never "// rest of code here"

action=show_images
  "give me 5 images of a BMW" / "photos of the Eiffel Tower" / "show me pictures of cats" / "image of X" / "pics of X"
  → HARD RULE: if the word "image(s)", "photo(s)", "picture(s)", or "pic(s)" appears ANYWHERE in the query, action MUST be show_images (or answer_with_images only if they also explicitly ask to explain/tell about something). NEVER show_results, NEVER plain answer.
  → answer="" always for show_images — no exceptions, no explanation text, images ONLY

action=show_results
  "give me a list of vegan restaurants" / "links about climate change" / "show me 10 results for react tutorials"
  "give me one result for the best pizza place" / "just the single best link for X" / "I'm feeling lucky, find me the top site for X"
  → if user asks for ONE / a single / the best / the top result, set quantity=1 — still action=show_results, just with quantity 1

action=answer
  "what is the capital of France" / "explain quantum computing" / "why is the sky blue" / "who won the 2022 world cup"
  → any question expecting a written explanation, no links or images needed

action=answer_with_images
  "tell me about the Eiffel Tower and show me a picture" / "explain how engines work with diagrams"
  → only when user explicitly wants BOTH an explanation AND visuals

Rules:
- quantity = EXACTLY {explicit_qty if explicit_qty is not None else 5}, unless the query itself implies "one/single/best/top" → then quantity=1
- NEVER change the subject matter of what the user asked
- when in doubt between show_results and answer, prefer answer only if the user is clearly asking a question (who/what/why/how/explain); prefer show_results if they're asking to find/give/show a thing/place/link"""

    try:
        text = ask_groq(prompt)
        start, end = text.find('{'), text.rfind('}')+1
        data = json.loads(text[start:end])
    except Exception as e:
        print(f"[understand fallback] groq/parse failed: {e}")
        data = {
            "action": "show_results",
            "understood": query,
            "search_query": query,
            "image_search_query": query,
            "direct_url": "",
            "quantity": explicit_qty or 5,
            "answer": "",
        }

    action = data.get("action","answer")
    if force_action in ("answer","show_images","show_results","answer_with_images"):
        action = force_action
    qty = max(1, min(explicit_qty or int(data.get("quantity",5)), 100))

    if action == "open_website":
        return {
            "action":"open_website","url":data.get("direct_url",""),
            "understood":data.get("understood",""),
            "answer":"","images":[],"results":[],"sources":[],"social":None,"wiki":None,"code":None,
        }

    if action == "go_to_thing":
        thing_q = data.get("thing_query") or data.get("search_query", query)
        hits = exa_search(thing_q, num=3, include_text=False)
        if hits:
            return {
                "action":"open_website","url":hits[0]["url"],
                "understood":data.get("understood",""),
                "answer":"","images":[],"results":[],"sources":[],"social":None,"wiki":None,"code":None,
            }
        # nothing found — fall back to a results list instead of a dead end
        action = "show_results"

    if action == "build_code":
        code_content = data.get("code_content", "")
        if not code_content.strip():
            # Groq failed to produce code — fall back gracefully instead of showing an empty container
            action = "answer"
            data["answer"] = "I wasn't able to generate that — try rephrasing what you'd like built."
        else:
            return {
                "action": "build_code",
                "url": "",
                "understood": data.get("understood",""),
                "answer": "",
                "images": [], "results": [], "sources": [], "social": None, "wiki": None,
                "code": {
                    "title": data.get("code_title", "Generated Code"),
                    "language": data.get("code_language", "text"),
                    "content": code_content,
                    "explainer": data.get("code_explainer", ""),
                },
            }

    images, results, sources = [], [], []
    answer = data.get("answer","")
    if action == "show_images":
        answer = ""
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
    if not results and action != "show_images":
        exa_n = qty if action == "show_results" else 5
        exa_results = exa_search(data.get("search_query", query), num=min(exa_n,10))
        results = exa_results[:qty] if action=="show_results" else exa_results[:4]

    exa_for_sources = exa_search(data.get("search_query",query), num=5) if (not results and action != "show_images") else results
    for item in exa_for_sources[:6]:
        dom = item.get("domain","")
        if dom and not any(s["domain"]==dom for s in sources):
            sources.append({"domain":dom,"url":item["url"]})

    # ── WIKIPEDIA enrichment (optional, never replaces the main answer) ──
    wiki = None
    if action in ["answer","answer_with_images"]:
        wiki = wikipedia_summary(data.get("search_query", query))

    return {
        "action": action,
        "url": "",
        "understood": data.get("understood",""),
        "answer": answer,
        "images": images,
        "results": results,
        "sources": sources,
        "social": social_meta,
        "wiki": wiki,
        "code": None,
    }
