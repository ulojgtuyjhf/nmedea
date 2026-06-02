import os
import json
import requests

# Retrieve keys directly from Render's Environment settings
GROQ_TOKEN = os.getenv("GROQ_TOKEN")
TAVILY_KEY = os.getenv("TAVILY_KEY")

def ask_groq_json(prompt, max_tokens=3000):
    """Queries Groq expecting a structured JSON response object."""
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_TOKEN}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"}
            },
            timeout=30
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response.")
            
        return json.loads(text[start:end])
    except Exception as e:
        # Prevent server crashes by returning a safe default structure
        return {
            "action": "answer",
            "understood": "Failed to parse system intent smoothly.",
            "search_query": prompt[:50],
            "image_search_query": "",
            "direct_url": "",
            "quantity": 5
        }

def ask_groq_to_synthesize(query, web_context):
    """Uses live web data context to write an accurate, up-to-date response."""
    prompt = f"""You are an advanced AI search assistant. 
User Query: "{query}"

Here is the live, up-to-date internet context found for this query:
{web_context}

Based on the live context above, write a COMPLETE, DETAILED, and thoroughly comprehensive response addressing the user's query. Rely strictly on the facts provided above."""
    
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_TOKEN}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 3000
            },
            timeout=30
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "Error synthesizing live search results cleanly."

def understand(query):
    """Main parsing and execution broker mapping user prompts to search actions."""
    intent_prompt = f"""You are the intent parsing engine for a powerful search tool. User said: "{query}"

Reply ONLY with a JSON object containing these keys:
{{
  "action": "answer OR open_website OR show_images OR show_results OR answer_with_images",
  "understood": "what user wants in one clear sentence",
  "search_query": "the perfect search engine query to find this information",
  "image_search_query": "specific image search query if images needed e.g. 'fast cars 4k photos', otherwise empty",
  "direct_url": "full URL if user explicitly wants to open a site, else empty string",
  "quantity": 5
}}

Rules:
- open/go to/visit site -> action=open_website, direct_url=full URL
- wants images/photos/pictures -> action=show_images
- wants images AND answer together -> action=answer_with_images
- asks a question / wants an explanation -> action=answer
- wants list/results/recommendations -> action=show_results
- quantity = EXACT number user requested, default 5"""

    data = ask_groq_json(intent_prompt)
    
    action = data.get("action", "answer")
    qty = int(data.get("quantity", 5))
    understood = data.get("understood", "")
    
    output_payload = {
        "action": action,
        "url": "",
        "understood": understood,
        "answer": "",
        "images": [],
        "results": [],
        "sources": []
    }

    if action == "open_website":
        output_payload["url"] = data.get("direct_url", "")
        return output_payload

    images, results, sources = [], [], []
    web_context_chunks = []

    # Gather search results or text intelligence data
    if action in ["show_results", "answer", "answer_with_images"]:
        try:
            tv_res = requests.post("https://api.tavily.com/search", json={
                "api_key": TAVILY_KEY,
                "query": data.get("search_query", query),
                "search_depth": "advanced",
                "include_answer": True,
                "max_results": qty if action == "show_results" else 5
            }, timeout=30)
            
            if tv_res.status_code == 200:
                tv_data = tv_res.json()
                
                if action == "show_results":
                    for r in tv_data.get("results", [])[:qty]:
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "description": r.get("content", "")[:300],
                            "image": r.get("image", "")
                        })
                
                for r in tv_data.get("results", []):
                    content = r.get("content", "")
                    web_context_chunks.append(f"Source: {r.get('url','')}\nContent: {content}\n---")
                    
                    try:
                        dom = r.get("url", "").split("/")[2].replace("www.", "")
                        if not any(s["domain"] == dom for s in sources):
                            sources.append({"domain": dom, "url": r.get("url", "")})
                    except:
                        pass
        except Exception:
            pass

    # Gather image assets
    if action in ["show_images", "answer_with_images"]:
        img_q = data.get("image_search_query") or data.get("search_query", query) + " photos high quality"
        try:
            tv_img = requests.post("https://api.tavily.com/search", json={
                "api_key": TAVILY_KEY,
                "query": img_q,
                "search_depth": "advanced",
                "include_images": True,
                "max_results": qty + 3
            }, timeout=30)
            
            if tv_img.status_code == 200:
                tv_img_data = tv_img.json()
                src_map = {s["domain"]: s["url"] for s in sources}
                
                for img in tv_img_data.get("images", []):
                    url = img if isinstance(img, str) else img.get("url", "")
                    if not url.startswith("http"): 
                        continue
                    
                    dom = ""
                    for d in src_map:
                        if d.lower() in url.lower():
                            dom = d
                            break
                    if not dom and sources:
                        dom = sources[min(len(images), len(sources)-1)]["domain"]
                    elif not dom:
                        dom = "web-image"
                        
                    images.append({"url": url, "source": dom})
                    if len(images) >= qty: 
                        break
        except Exception:
            pass

    # Assemble everything into the exact return structure required by your app
    if action in ["answer", "answer_with_images"] and web_context_chunks:
        context_string = "\n".join(web_context_chunks)
        output_payload["answer"] = ask_groq_to_synthesize(query, context_string)

    output_payload["images"] = images
    output_payload["results"] = results
    output_payload["sources"] = sources

    return output_payload
