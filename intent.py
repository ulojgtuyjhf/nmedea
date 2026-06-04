import requests, json, os, re
from dotenv import load_dotenv
load_dotenv()

GROQ_TOKEN = os.getenv("GROQ_TOKEN")
TAVILY_KEY = os.getenv("TAVILY_KEY")


def ask_groq(prompt, max_tokens=3000):
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


def extract_quantity_from_query(query: str) -> int | None:
    """
    Scan the raw query for explicit number mentions so we honour
    things like 'show me 1 image', 'give me 10 pictures', 'list 7 results'.
    Returns None if no number found (caller uses default).
    """
    # written numbers → digits
    word_map = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    }
    q = query.lower()
    for word, num in word_map.items():
        if re.search(r'\b' + word + r'\b', q):
            return num

    # digit patterns: "3 images", "show 5", "give me 8 photos", etc.
    m = re.search(
        r'\b(\d{1,2})\s*(?:image|photo|picture|pic|result|link|item|thing|show|news|article)s?\b',
        q
    )
    if m:
        return int(m.group(1))

    # bare digit anywhere in short queries like "show me 4"
    m2 = re.search(r'\bgive\s+me\s+(\d{1,2})\b', q)
    if m2:
        return int(m2.group(1))

    return None


def understand(query):
    # ── 1. Extract explicit quantity before sending to Groq ──
    explicit_qty = extract_quantity_from_query(query)

    prompt = f"""You are the world's most powerful AI search engine. User said: "{query}"

Reply ONLY this JSON, nothing else:
{{
  "action": "answer OR open_website OR show_images OR show_results OR answer_with_images",
  "understood": "what user wants in one clear sentence",
  "search_query": "perfect, highly specific search query that matches EXACTLY what the user asked for",
  "image_search_query": "very specific image search query if images needed — be precise about the subject, gender, age, style e.g. 'male basketball players dunking 4k', NOT just the topic",
  "direct_url": "full URL if user wants to open a site, else empty string",
  "quantity": {explicit_qty if explicit_qty is not None else 5},
  "answer": "if action is answer or answer_with_images: write COMPLETE DETAILED response with no word limit. else empty string"
}}

Rules:
- open/go to/visit → action=open_website, direct_url=full URL
- wants images/photos/pictures → action=show_images, answer must be empty string
- wants images AND information → action=answer_with_images
- asks a question → action=answer with FULL unlimited answer
- wants list/results/recommendations → action=show_results
- quantity = EXACTLY {explicit_qty if explicit_qty is not None else "use 5 as default"}
- For image_search_query: be VERY specific — include gender, age, style, quality descriptors so the right images are returned. If user says "boys" use "boys", never substitute."""

    text = ask_groq(prompt)
    start = text.find('{')
    end = text.rfind('}') + 1
    data = json.loads(text[start:end])

    action = data.get("action", "answer")
    # Honour explicit quantity; Groq may still override to something wrong
    qty = explicit_qty if explicit_qty is not None else int(data.get("quantity", 5))
    # Clamp to sane range
    qty = max(1, min(qty, 20))

    if action == "open_website":
        return {
            "action": "open_website",
            "url": data.get("direct_url", ""),
            "understood": data.get("understood", ""),
            "answer": "", "images": [], "results": [], "sources": []
        }

    images, results, sources = [], [], []
    answer = data.get("answer", "")

    # ── 2. FETCH IMAGES ──
    if action in ["show_images", "answer_with_images"]:
        img_q = data.get("image_search_query") or data.get("search_query", query) + " high quality photos"
        # Request more than needed so we can filter bad ones
        fetch_count = qty + 10
        try:
            tv = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_KEY,
                    "query": img_q,
                    "search_depth": "advanced",
                    "include_images": True,
                    "include_image_descriptions": True,
                    "max_results": fetch_count
                },
                timeout=30
            ).json()

            src_map = {}
            for r in tv.get("results", []):
                try:
                    dom = r.get("url", "").split("/")[2].replace("www.", "")
                    src_map[dom] = r.get("url", "")
                    if not any(s["domain"] == dom for s in sources):
                        sources.append({"domain": dom, "url": r.get("url", "")})
                except Exception:
                    pass

            raw_imgs = tv.get("images", [])
            for img in raw_imgs:
                url = img if isinstance(img, str) else img.get("url", "")
                if not url.startswith("http"):
                    continue
                # derive source domain
                dom = ""
                for d in src_map:
                    if d.lower() in url.lower():
                        dom = d
                        break
                if not dom and sources:
                    idx = min(len(images), len(sources) - 1)
                    dom = sources[idx]["domain"]
                images.append({"url": url, "source": dom})
                if len(images) >= qty:
                    break

        except Exception as e:
            print(f"[image fetch error] {e}")

    # ── 3. FETCH WEB RESULTS / CONTEXT ──
    if action in ["show_results", "answer", "answer_with_images"]:
        fetch_n = qty if action == "show_results" else 3
        try:
            tv = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_KEY,
                    "query": data.get("search_query", query),
                    "search_depth": "advanced",
                    "include_answer": True,
                    "max_results": fetch_n
                },
                timeout=30
            ).json()

            if action == "show_results":
                for r in tv.get("results", [])[:qty]:
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "description": r.get("content", "")[:300],
                        "image": r.get("image", "")
                    })

            for r in tv.get("results", [])[:5]:
                try:
                    dom = r.get("url", "").split("/")[2].replace("www.", "")
                    if not any(s["domain"] == dom for s in sources):
                        sources.append({"domain": dom, "url": r.get("url", "")})
                except Exception:
                    pass

        except Exception as e:
            print(f"[results fetch error] {e}")

    return {
        "action": action,
        "url": "",
        "understood": data.get("understood", ""),
        "answer": answer,
        "images": images,
        "results": results,
        "sources": sources
    }
