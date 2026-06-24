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
#  GROQ VISION  — understands an uploaded/captured photo
# ─────────────────────────────────────────────────────────────────
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # vision-capable Groq model

def ask_groq_vision(image_base64: str, image_mime: str, user_text: str = ""):
    """Send an image (as base64 data URL) + optional user question to a vision-capable
    Groq model and get back a natural-language description/answer."""
    data_url = f"data:{image_mime or 'image/jpeg'};base64,{image_base64}"

    if user_text and user_text.strip():
        prompt_text = (
            f"The user uploaded this image and asked: \"{user_text.strip()}\". "
            "Answer their question about the image directly and naturally. "
            "If they didn't really ask anything specific, just describe what's in the image "
            "clearly and usefully, as if explaining it to someone who can't see it. "
            "Keep it to 2-4 short sentences unless more detail is clearly needed."
        )
    else:
        prompt_text = (
            "Describe what's in this image clearly and usefully, as if explaining it to "
            "someone who can't see it. Mention the main subject, relevant details, and "
            "anything notable (text, brand, location clues, etc). Keep it to 2-4 short sentences."
        )

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_TOKEN}"},
        json={
            "model": GROQ_VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            "max_tokens": 600,
        },
        timeout=30,
    )
    data = r.json()
    if "choices" not in data:
        print(f"[groq vision error] status={r.status_code} body={data}")
        raise RuntimeError(f"Groq vision API error: {data.get('error', data)}")
    return data["choices"][0]["message"]["content"].strip()


def ask_groq_vision_keywords(image_base64: str, image_mime: str):
    """Ask the vision model for a short, search-friendly phrase describing the image,
    so we can run a normal web search alongside the description (e.g. 'golden retriever
    dog breed', 'Eiffel Tower Paris', 'Nike Air Max sneaker'). Best-effort — returns ''
    on failure rather than raising, since this is supplementary."""
    data_url = f"data:{image_mime or 'image/jpeg'};base64,{image_base64}"
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_TOKEN}"},
            json={
                "model": GROQ_VISION_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "Reply with ONLY a short, specific search-engine query (3-7 words) "
                            "that identifies the main subject of this image — e.g. 'golden "
                            "retriever dog breed', 'Eiffel Tower Paris France', 'iPhone 15 Pro "
                            "specs'. No punctuation, no explanation, just the query."
                        )},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                "max_tokens": 30,
            },
            timeout=20,
        )
        data = r.json()
        if "choices" not in data:
            return ""
        return data["choices"][0]["message"]["content"].strip().strip('"')
    except Exception as e:
        print(f"[groq vision keywords error] {e}")
        return ""


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


CODE_LANGS = [
    "python","javascript","js","java","c\\+\\+","cpp","c#","csharp","html","css",
    "typescript","ts","php","ruby","go","golang","rust","swift","kotlin","sql","bash","shell",
]
def looks_like_code_request(query: str) -> bool:
    """Deterministic backup signal — if this is clearly a code request,
    we don't want to depend solely on Groq classifying it correctly."""
    q = query.lower()
    has_lang = any(re.search(r'\b' + lang + r'\b', q) for lang in CODE_LANGS)
    has_code_word = re.search(r'\b(script|function|program|algorithm|code|app|webpage|website|class|api endpoint)\b', q)
    has_verb = re.search(r'\b(write|build|code|create|make|give me|generate)\b', q)
    if has_lang and has_verb:
        return True
    if has_code_word and has_verb:
        return True
    return False


# ─────────────────────────────────────────────────────────────────
#  CAMERA / UPLOAD INTENT  — purely typed, no buttons anywhere in the UI.
#  These are deterministic regex checks (not Groq) since the phrasing is
#  simple and unambiguous, and we don't want an LLM round-trip on the
#  common case of "open the camera" / "let me upload a photo".
# ─────────────────────────────────────────────────────────────────
def looks_like_camera_request(query: str) -> bool:
    q = query.lower()
    return bool(re.search(
        r'\b(open|launch|turn on|start|activate|show me|use)\b.{0,15}\bcamera\b'
        r'|\bcamera\b.{0,15}\b(open|please|now)\b'
        r'|^\s*camera\s*$',
        q,
    ))

def looks_like_upload_request(query: str) -> bool:
    q = query.lower()
    return bool(re.search(
        r'\b(upload|attach)\b.{0,25}\b(image|photo|picture|pic)\b'
        r'|\b(image|photo|picture|pic)\b.{0,15}\bupload\b'
        r'|\b(give|show)\s+me\b.{0,25}\b(an?\s+)?upload\b'
        r'|\bupload\b.{0,15}\binterface\b'
        r'|\binterface\b.{0,15}\bupload\b'
        r'|\blet\s+me\s+upload\b',
        q,
    ))

def looks_like_todo_request(query: str) -> bool:
    q = query.lower()
    return bool(re.search(
        r'\b(give|show|make|create|build|start)\s+me\b.{0,15}\b(a|my)?\s*to[\s-]?do\s*list\b'
        r'|\bto[\s-]?do\s*list\b'
        r'|\bcreate\s+a\s+checklist\b'
        r'|\bmake\s+(me\s+)?a\s+checklist\b',
        q,
    ))

def looks_like_music_request(query: str) -> bool:
    """Deliberately requires an explicit music/song/lyric phrase — never
    triggers on a bare word or title alone (e.g. just 'dandelion'), since
    that would false-positive on every short factual query. The person
    needs to say it's music they're after."""
    q = query.lower()
    return bool(re.search(
        r'\b(song|music|track|lyric|lyrics)\b.{0,20}\b(that|which|called|named|goes|says|sounds like)\b'
        r'|\b(give|show|find|play)\s+me\b.{0,20}\b(that|the|a)\b.{0,20}\b(song|music|track)\b'
        r'|\bwhat\s+(is\s+)?(this|that)\s+song\b'
        r'|\bwhat\s+song\s+(is\s+)?(this|that)\b'
        r'|\bidentify\s+(this|that)\s+song\b'
        r'|\b(name|title)\s+of\s+(this|that)\s+song\b'
        r'|\bsong\s+that\s+goes\b',
        q,
    ))

def looks_like_weather_request(query: str) -> bool:
    q = query.lower()
    return bool(re.search(
        r'\bweather\b'
        r'|\bforecast\b'
        r'|\bis\s+it\s+(going\s+to\s+)?(rain|raining|snow|snowing|sunny|cold|hot)\b'
        r'|\btemperature\s+(today|tomorrow|outside|right\s+now)\b',
        q,
    ))

def looks_like_calculator_request(query: str) -> bool:
    q = query.lower()
    return bool(re.search(
        r'\b(give|show|open|need|want)\b.{0,15}\bcalculator\b'
        r'|^\s*calculator\s*$'
        r'|\bopen\s+(a\s+|the\s+)?calculator\b'
        r'|\bcalculator\s+(app|please|now)\b',
        q,
    ))


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
    """Fetch up to `qty` images. Makes multiple Tavily calls if qty > 20.

    Bug fix: earlier versions appended generic words ("hd photos", "gallery",
    etc) to the query on later calls to fetch more images. That drifts the
    search away from what the person actually asked for — e.g. asking for
    30 images of "boys playing basketball" could start pulling generic
    "basketball gallery" stock photos by the 3rd/4th call that have nothing
    to do with the original subject. Every call now uses the EXACT same
    query; duplicate URLs across calls are already deduped below, so this
    costs nothing and keeps every batch on-topic.
    """
    all_images = []
    seen_urls = set()
    batch = min(qty + 10, 30)  # Tavily max per call

    # Run up to 5 calls to reach high qty (e.g. 100)
    calls_needed = min(5, (qty + batch - 1) // max(batch - 10, 1))
    for call_i in range(calls_needed):
        if len(all_images) >= qty:
            break
        try:
            tv = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": TAVILY_KEY, "query": query,
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
#  MUSIC IDENTIFICATION  — iTunes Search API (free, no key, no auth)
#  Per Apple's terms, previews/artwork are for promotional/identification
#  purposes only — streamed, never downloaded — paired with a real store
#  link. This mirrors how Shazam/Google "identify this song" results work:
#  title, artist, artwork, a 30-second preview, and links to actually
#  listen on a licensed platform. No full track audio, ever.
# ─────────────────────────────────────────────────────────────────
def music_search(query: str, limit: int = 5):
    """Look up songs matching a name, lyric snippet, or description.
    Returns a list of track dicts, or an empty list if nothing matched —
    caller should treat an empty list as 'couldn't identify that song',
    not as an error."""
    try:
        r = requests.get(
            "https://itunes.apple.com/search",
            params={"term": query, "media": "music", "entity": "song", "limit": limit},
            timeout=10,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        tracks = []
        for item in data.get("results", []):
            if item.get("wrapperType") != "track":
                continue
            track_name = item.get("trackName", "")
            artist_name = item.get("artistName", "")
            if not track_name or not artist_name:
                continue
            search_term = requests.utils.quote(f"{track_name} {artist_name}")
            tracks.append({
                "title": track_name,
                "artist": artist_name,
                "album": item.get("collectionName", ""),
                "artwork": (item.get("artworkUrl100") or item.get("artworkUrl60") or "").replace("100x100", "300x300"),
                "preview_url": item.get("previewUrl", ""),  # 30-second licensed preview, stream-only
                "itunes_url": item.get("trackViewUrl", ""),
                "spotify_search_url": f"https://open.spotify.com/search/{search_term}",
                "youtube_search_url": f"https://www.youtube.com/results?search_query={search_term}",
                "release_date": (item.get("releaseDate") or "")[:10],
            })
        return tracks
    except Exception as e:
        print(f"[music_search error] {e}")
        return []


# ─────────────────────────────────────────────────────────────────
#  WEATHER  — Open-Meteo (free, no key, no auth)
# ─────────────────────────────────────────────────────────────────
WEATHER_CODE_MAP = {
    0: "Clear sky", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Freezing fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle",
    56: "Freezing drizzle", 57: "Dense freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Heavy freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light showers", 81: "Showers", 82: "Violent showers",
    85: "Light snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm with hail",
}

def geocode_place(place: str):
    """Resolve a place name to lat/lon using Open-Meteo's free geocoding endpoint."""
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place, "count": 1},
            timeout=8,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("results")
        if not results:
            return None
        top = results[0]
        label_parts = [top.get("name", "")]
        if top.get("admin1"):
            label_parts.append(top["admin1"])
        if top.get("country"):
            label_parts.append(top["country"])
        return {
            "lat": top["latitude"], "lon": top["longitude"],
            "label": ", ".join(p for p in label_parts if p),
        }
    except Exception as e:
        print(f"[geocode error] {e}")
        return None

def weather_lookup(place: str = None, lat: float = None, lon: float = None):
    """Fetch current conditions + a short daily forecast for a place name
    OR explicit coordinates. Returns None if the place can't be resolved
    or the weather API fails — caller should treat that as 'try again
    with a different location', not crash the whole response."""
    label = None
    if (lat is None or lon is None) and place:
        geo = geocode_place(place)
        if not geo:
            return None
        lat, lon, label = geo["lat"], geo["lon"], geo["label"]
    if lat is None or lon is None:
        return None
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "timezone": "auto",
                "forecast_days": 5,
            },
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        current = data.get("current", {})
        daily = data.get("daily", {})
        code = current.get("weather_code")
        days = []
        dates = daily.get("time", [])
        for i in range(min(5, len(dates))):
            d_code = daily.get("weather_code", [None]*5)[i]
            days.append({
                "date": dates[i],
                "max": daily.get("temperature_2m_max", [None]*5)[i],
                "min": daily.get("temperature_2m_min", [None]*5)[i],
                "condition": WEATHER_CODE_MAP.get(d_code, ""),
            })
        return {
            "label": label or f"{lat:.2f}, {lon:.2f}",
            "temp": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "condition": WEATHER_CODE_MAP.get(code, ""),
            "daily": days,
        }
    except Exception as e:
        print(f"[weather error] {e}")
        return None


# ─────────────────────────────────────────────────────────────────
#  IMAGE UNDERSTANDING  — handles an uploaded/captured photo end-to-end
# ─────────────────────────────────────────────────────────────────
def understand_image(image_base64: str, image_mime: str, user_text: str = ""):
    """Given a base64-encoded image (and optional accompanying question/text),
    returns a dict with a natural-language description plus supporting web
    results, in the same shape the frontend's `renderResults()` expects:
      { action, understood, answer, images, results, sources, social, wiki,
        code, visual: { description, thumbnail } }
    Never raises — on failure it returns a graceful fallback so the UI still
    has something sensible to show.
    """
    thumbnail = f"data:{image_mime or 'image/jpeg'};base64,{image_base64}"

    try:
        description = ask_groq_vision(image_base64, image_mime, user_text)
    except Exception as e:
        print(f"[understand_image vision error] {e}")
        return {
            "action": "show_results",
            "understood": "Understand the uploaded image",
            "answer": "", "images": [], "results": [], "sources": [],
            "social": None, "wiki": None, "code": None,
            "visual": {
                "description": "I couldn't quite analyze that image — give it another try, or add a question about what you'd like to know.",
                "thumbnail": thumbnail,
            },
        }

    # Best-effort companion web search so the person also gets real links,
    # e.g. if it's a dog breed, a landmark, a product, etc.
    search_q = (user_text.strip() if user_text and user_text.strip() else "") or ask_groq_vision_keywords(image_base64, image_mime)
    results, sources = [], []
    if search_q:
        try:
            hits = exa_search(search_q, num=5)
            results = hits[:4]
            for item in hits[:6]:
                dom = item.get("domain", "")
                if dom and not any(s["domain"] == dom for s in sources):
                    sources.append({"domain": dom, "url": item["url"]})
        except Exception as e:
            print(f"[understand_image search error] {e}")

    return {
        "action": "show_results",
        "understood": user_text.strip() if user_text and user_text.strip() else "Understand the uploaded image",
        "answer": "",
        "images": [],
        "results": results,
        "sources": sources,
        "social": None,
        "wiki": None,
        "code": None,
        "visual": {
            "description": description,
            "thumbnail": thumbnail,
        },
    }


# ─────────────────────────────────────────────────────────────────
#  MAIN  understand()
# ─────────────────────────────────────────────────────────────────
def understand(query: str, force_action: str = None, image_base64: str = None, image_mime: str = None, lat: float = None, lon: float = None):
    # ── VISUAL SEARCH: an image was attached — handle via vision model, skip the text pipeline ──
    if image_base64:
        return understand_image(image_base64, image_mime, query or "")

    # ── TYPED CAMERA / UPLOAD INTENT — no buttons in the UI, this is the only entry point.
    #    Handled deterministically (no Groq round-trip) and skips the rest of the pipeline,
    #    since the frontend renders these as their own inline card. ──
    if looks_like_camera_request(query):
        return {
            "action": "open_camera",
            "url": "",
            "understood": "Open the camera",
            "answer": "", "images": [], "results": [], "sources": [],
            "social": None, "wiki": None, "code": None,
        }
    if looks_like_upload_request(query):
        return {
            "action": "show_upload",
            "url": "",
            "understood": "Show an image upload interface",
            "answer": "", "images": [], "results": [], "sources": [],
            "social": None, "wiki": None, "code": None,
        }
    if looks_like_todo_request(query):
        return {
            "action": "show_todo",
            "url": "",
            "understood": "Show a to-do list",
            "answer": "", "images": [], "results": [], "sources": [],
            "social": None, "wiki": None, "code": None,
        }
    if looks_like_music_request(query):
        # Strip wrapper phrasing ("give me that song that says...", "what
        # is this song called", etc) down to the actual title/lyric/artist
        # to search for. If nothing meaningful survives the strip (e.g. the
        # person never actually named the song — "what song is this?" on
        # its own), there's nothing to search for yet.
        clean = re.sub(
            r'\b(give|show|find|play)\s+me\b'
            r'|\b(that|this|the|a|an)\b'
            r'|\b(song|music|track)\b'
            r'|\b(goes|says|sounds\s+like|called|named|which)\b'
            r"|\bwhat'?s?\b|\bwhat\s+is\b|\bis\b"
            r'|\bidentify\b|\bname\s+of\b|\btitle\s+of\b',
            ' ', query, flags=re.IGNORECASE,
        )
        clean = re.sub(r'\s+', ' ', clean).strip(' ?.,!')
        if not clean:
            return {
                "action": "answer",
                "url": "",
                "understood": "Identify a song",
                "answer": "Tell me a lyric, the title, or the artist and I'll find it for you.",
                "images": [], "results": [], "sources": [],
                "social": None, "wiki": None, "code": None,
            }
        tracks = music_search(clean)
        return {
            "action": "show_music",
            "url": "",
            "understood": f"Find the song: {clean}",
            "answer": "", "images": [], "results": [], "sources": [],
            "social": None, "wiki": None, "code": None,
            "music": tracks,
        }
    if looks_like_weather_request(query):
        # Pull a place name out if one was given (e.g. "weather in Tokyo").
        # If none was given, fall back to lat/lon supplied by the frontend
        # (browser geolocation) — and if that's not available either,
        # signal that a location is needed instead of guessing one.
        place = None
        m = re.search(r'\b(?:in|for|at)\s+([a-zA-Z][a-zA-Z\s,.\-]{1,60})$', query.strip(), re.IGNORECASE)
        if m:
            place = m.group(1).strip(' ?.,!')
        if place:
            weather = weather_lookup(place=place)
        elif lat is not None and lon is not None:
            weather = weather_lookup(lat=lat, lon=lon)
        else:
            weather = None
        return {
            "action": "show_weather",
            "url": "",
            "understood": f"Weather for {place}" if place else "Weather for your location",
            "answer": "", "images": [], "results": [], "sources": [],
            "social": None, "wiki": None, "code": None,
            "weather": weather,
            "weather_needs_location": weather is None,
        }
    if looks_like_calculator_request(query):
        return {
            "action": "show_calculator",
            "url": "",
            "understood": "Show a calculator",
            "answer": "", "images": [], "results": [], "sources": [],
            "social": None, "wiki": None, "code": None,
        }

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
  "write me a python script that..." / "code a to-do list app" / "build a snake game in javascript" / "give me HTML for a landing page" / "create a function that sorts an array" / "reverse a string in python" / "python script to reverse a string"
  → HARD RULE: if the query mentions a programming language (python, javascript, java, c++, html, css, etc.) OR words like "script", "function", "code", "program", "app" combined with "write/build/create/give me/make", action MUST be build_code. NEVER plain answer for these — even if the request sounds like a simple one-liner.
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

    # SAFETY NET: Groq sometimes misses obvious code requests — catch it deterministically
    if action not in ("build_code","show_images") and looks_like_code_request(query):
        code_prompt = f"""The user asked: "{query}"

This is a request for source code. Reply ONLY this JSON, nothing else:
{{
  "code_title": "a short 3-6 word title for what the code does",
  "code_language": "the language as a lowercase string, e.g. python, javascript, html",
  "code_content": "the COMPLETE, working, runnable code — no placeholders, no '...rest of code', no TODOs",
  "code_explainer": "a short 2-3 sentence explanation of what the code does and how to use/run it"
}}"""
        try:
            code_text = ask_groq(code_prompt)
            cs, ce = code_text.find('{'), code_text.rfind('}')+1
            code_data = json.loads(code_text[cs:ce])
            if code_data.get("code_content","").strip():
                action = "build_code"
                data.update(code_data)
        except Exception as e:
            print(f"[code safety-net failed] {e}")
            # leave action as whatever Groq originally said — graceful, no crash

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
