"""
Security Platform - Flask backend (app.py)

Auth (signup/login) now happens directly in the browser via Firebase Auth —
this backend no longer handles passwords at all. It only does two things
that must stay server-side:

  1. API key creation/management — generating and hashing a real secret
     key has to happen on a server, never in browser JS.
  2. The /api/check detection endpoint — the actual rule logic. If this
     ran in the browser, anyone could read the thresholds and craft
     requests that dodge them, or just call "allow" themselves.

Every protected route here verifies a Firebase ID token (sent from the
frontend after login) using the Firebase Admin SDK — this confirms the
request really came from a logged-in user, without this backend ever
touching a password.
"""

import hashlib
import os
import re
import secrets
import time
import uuid

import firebase_admin
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials, db
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Firebase setup (Admin SDK — server-side only, uses the service account key)
# ---------------------------------------------------------------------------
FIREBASE_DATABASE_URL = os.environ.get(
    "FIREBASE_DATABASE_URL",
    "https://globalchat-2d669-default-rtdb.firebaseio.com",
)


def init_firebase():
    if firebase_admin._apps:
        return

    key_json = os.environ.get("FIREBASE_KEY_JSON")
    if key_json:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write(key_json)
        tmp.close()
        cred = credentials.Certificate(tmp.name)
    else:
        local_path = os.path.join(os.path.dirname(__file__), "firebase-key.json")
        cred = credentials.Certificate(local_path)

    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})


def ref(path):
    init_firebase()
    return db.reference(path)


# ---------------------------------------------------------------------------
# Auth — verify the Firebase ID token sent from the frontend
# ---------------------------------------------------------------------------
def require_auth():
    """Returns the user's Firebase UID if the token is valid, else None."""
    init_firebase()
    auth_header = request.headers.get("Authorization", "")
    id_token = auth_header.replace("Bearer ", "").strip()
    if not id_token:
        return None
    try:
        decoded = firebase_auth.verify_id_token(id_token)
        return decoded["uid"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------
def generate_api_key():
    return "sk_live_" + secrets.token_hex(24)


def hash_key(raw_key):
    return hashlib.sha256(raw_key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Detection engine (pattern-based, no LLM calls)
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = re.compile(
    r"(\bor\b\s+1\s*=\s*1)|union\s+select|drop\s+table|<script.*?>|javascript:|;\s*--|\bexec(\s|\()",
    re.IGNORECASE,
)

BRUTE_FORCE_WINDOW = 120
BRUTE_FORCE_MAX = 10
BOT_WINDOW = 60
BOT_MAX = 30


def log_event(key_id, ip, event_type):
    ip_safe = ip.replace(".", "_").replace(":", "_")
    ref(f"events/{key_id}/{ip_safe}/{event_type}").push(time.time())


def count_recent(key_id, ip, event_type, window):
    ip_safe = ip.replace(".", "_").replace(":", "_")
    data = ref(f"events/{key_id}/{ip_safe}/{event_type}").get() or {}
    cutoff = time.time() - window
    return len([t for t in data.values() if isinstance(t, (int, float)) and t >= cutoff])


def run_detection(key_id, ip, target, payload):
    if payload and INJECTION_PATTERNS.search(payload):
        return {"action": "block", "detector": "injection", "reason": "Suspicious input pattern detected"}

    log_event(key_id, ip, f"brute_{target}")
    brute_count = count_recent(key_id, ip, f"brute_{target}", BRUTE_FORCE_WINDOW)
    if brute_count > BRUTE_FORCE_MAX:
        return {
            "action": "block",
            "detector": "brute_force",
            "reason": f"{brute_count} attempts on '{target}' in {BRUTE_FORCE_WINDOW}s",
        }

    log_event(key_id, ip, "request")
    bot_count = count_recent(key_id, ip, "request", BOT_WINDOW)
    if bot_count > BOT_MAX:
        return {
            "action": "block",
            "detector": "bot",
            "reason": f"{bot_count} requests in {BOT_WINDOW}s",
        }

    return {"action": "allow", "detector": None, "reason": None}


# ---------------------------------------------------------------------------
# Routes — frontend
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — API keys (creation must stay server-side; everything else the
# frontend can read directly from Firebase using Security Rules)
# ---------------------------------------------------------------------------
@app.route("/api/keys/create", methods=["POST"])
def create_key():
    user_id = require_auth()
    if not user_id:
        return jsonify({"error": "Not authenticated."}), 401

    raw_key = generate_api_key()
    key_id = str(uuid.uuid4())

    ref(f"api_keys/{key_id}").set({
        "user_id": user_id,
        "key_hash": hash_key(raw_key),
        "active": True,
        "created_at": time.time(),
        "label": "Default key",
    })
    # Also store a pointer under the user's own UID so the frontend can
    # find their key(s) directly via Security Rules without this backend.
    ref(f"user_keys/{user_id}/{key_id}").set(True)

    return jsonify({"key_id": key_id, "api_key": raw_key})


@app.route("/api/keys/<key_id>/toggle", methods=["POST"])
def toggle_key(key_id):
    user_id = require_auth()
    if not user_id:
        return jsonify({"error": "Not authenticated."}), 401

    data = request.get_json(force=True)
    key_data = ref(f"api_keys/{key_id}").get()

    if not key_data or key_data.get("user_id") != user_id:
        return jsonify({"error": "API key not found."}), 404

    ref(f"api_keys/{key_id}").update({"active": bool(data.get("active"))})
    return jsonify({"key_id": key_id, "active": bool(data.get("active"))})


# ---------------------------------------------------------------------------
# Routes — status / dashboard data
# ---------------------------------------------------------------------------
@app.route("/api/status/<key_id>", methods=["GET"])
def get_status(key_id):
    user_id = require_auth()
    if not user_id:
        return jsonify({"error": "Not authenticated."}), 401

    key_data = ref(f"api_keys/{key_id}").get()
    if not key_data or key_data.get("user_id") != user_id:
        return jsonify({"error": "API key not found."}), 404

    logs = ref(f"logs/{key_id}").get() or {}
    recent = sorted(logs.values(), key=lambda e: e.get("timestamp", 0), reverse=True)[:20]

    return jsonify({
        "key_id": key_id,
        "active": key_data.get("active", True),
        "recent_events": recent,
        "total_events_logged": len(logs),
    })


# ---------------------------------------------------------------------------
# Routes — the actual detection endpoint customer sites call
# ---------------------------------------------------------------------------
@app.route("/api/check", methods=["POST"])
def check():
    raw_key = request.headers.get("x-api-key", "")
    if not raw_key:
        return jsonify({"error": "Missing x-api-key header."}), 401

    key_hash = hash_key(raw_key)
    all_keys = ref("api_keys").get() or {}
    key_id, key_data = None, None
    for kid, k in all_keys.items():
        if k.get("key_hash") == key_hash:
            key_id, key_data = kid, k
            break

    if not key_id:
        return jsonify({"error": "Invalid API key."}), 401

    if not key_data.get("active", True):
        return jsonify({"action": "allow", "reason": "Detection is paused for this key."})

    data = request.get_json(force=True)
    ip = data.get("ip", request.remote_addr or "0.0.0.0")
    target = data.get("target", "default")
    payload = data.get("payload", "")

    verdict = run_detection(key_id, ip, target, payload)

    if verdict["action"] in ("block", "flag"):
        ref(f"logs/{key_id}").push({
            "ip": ip,
            "target": target,
            "action": verdict["action"],
            "reason": verdict["reason"],
            "detector": verdict["detector"],
            "timestamp": time.time(),
        })

    return jsonify(verdict)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
