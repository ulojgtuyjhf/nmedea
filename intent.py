import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_TOKEN = os.getenv("GROQ_TOKEN")
TAVILY_KEY = os.getenv("TAVILY_KEY")

def ask_groq_json(prompt, max_tokens=3000):
    """Safely queries Groq expecting a JSON response."""
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_TOKEN}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                # Force JSON mode to prevent Groq from returning conversational text outside the markdown
                "response_format": {"type": "json_object"} 
            },
            timeout=30
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        
        # Robust JSON extraction
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in LLM response.")
            
        return json.loads(text[start:end])
    except Exception as e:
        print(f"[Groq Error]: {e}")
        # Fallback safe dictionary so your app doesn't crash
        return {
            "action": "answer",
            "understood": "Failed to parse intent.",
            "search_query": prompt[:50],
            "image_search_query": "",
            "direct_url": "",
            "quantity": 5
        }

def ask_groq_to_synthesize(query, web_context):
    """Uses web data to write a highly accurate, real-time answer."""
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
        print(f"[Synthesis Error]: {e}")
        return "Error synthesizing live search results."

def understand(query):
    # Step 1: Query the LLM purely to understand INTENT and get perfect search queries
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
    
    # Initialize your exact return structure
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

    # Step 2: Fetch Web Context / Results if needed
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
                
                # Build the results list if user wanted list/recommendations
                if action == "show_results":
                    for r in tv_data.get("results", [])[:qty]:
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "description": r.get("content", "")[:300],
                            "image": r.get("image", "")
                        })
                
                # Build up sources and aggregate text context for generation
                for r in tv_data.get("results", []):
                    content = r.get("content", "")
                    web_context_chunks.append(f"Source: {r.get('url','')}\nContent: {content}\n---")
                    
                    try:
                        dom = r.get("url", "").split("/")[2].replace("www.", "")
                        if not any(s["domain"] == dom for s in sources):
                            sources.append({"domain": dom, "url": r.get("url", "")})
                    except:
                        pass
        except Exception as e:
            print(f"[Tavily Web Search Error]: {e}")

    # Step 3: Fetch Images if requested
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
                
                # Pre-map domains from current sources for matching
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
        except Exception as e:
            print(f"[Tavily Image Search Error]: {e}")

    # Step 4: Generate a factual Answer based on the gathered Web Context
    if action in ["answer", "answer_with_images"] and web_context_chunks:
        context_string = "\n".join(web_context_chunks)
        output_payload["answer"] = ask_groq_to_synthesize(query, context_string)

    # Assign final values back to payload
    output_payload["images"] = images
    output_payload["results"] = results
    output_payload["sources"] = sources

    return output_payload
