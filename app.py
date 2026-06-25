"""
Security Platform - Flask backend (app.py)

Run locally:  python app.py
Deploy on Render: gunicorn app:app  (see Procfile)

Routes:
  GET  /                      -> serves the dashboard (templates/index.html)
  POST /api/signup            -> create account
  POST /api/login             -> log in, get a token
  POST /api/keys/create       -> create a new API key (auth required)
  GET  /api/keys              -> list your API keys (auth required)
  POST /api/keys/<id>/toggle  -> pause/resume a key (auth required)
  GET  /api/status/<id>       -> live status + recent activity for a key (auth required)
  POST /api/check             -> the detection endpoint customer sites call (uses x-api-key header)
"""

import hashlib
import os
import re
import secrets
import time
import uuid

import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, jsonify, render_template, request
from passlib.hash import bcrypt

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Firebase setup
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
        import json
        import tempfile

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write(key_json)
        tmp.close()
        cred = credentials.Certificate(tmp.name)
    else:
        # Local dev: expects firebase-key.json next to this file
        local_path = os.path.join(os.path.dirname(__file__), "firebase-key.json")
        cred = credentials.Certificate(local_path)

    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})


def ref(path):
    init_firebase()
    return db.reference(path)


# ---------------------------------------------------------------------------
# Simple session tokens (no JWT library needed — random token stored in DB)
# ---------------------------------------------------------------------------
def make_token():
    return secrets.token_hex(32)


def get_user_from_token(token):
    if not token:
        return None
    session = ref(f"sessions/{token}").get()
    if not session:
        return None
    return session.get("user_id")


def require_auth():
    """Returns user_id if valid, or None if not authenticated."""
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "").strip()
    return get_user_from_token(token)


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
    # 1. Injection check
    if payload and INJECTION_PATTERNS.search(payload):
        return {"action": "block", "detector": "injection", "reason": "Suspicious input pattern detected"}

    # 2. Brute-force check
    log_event(key_id, ip, f"brute_{target}")
    brute_count = count_recent(key_id, ip, f"brute_{target}", BRUTE_FORCE_WINDOW)
    if brute_count > BRUTE_FORCE_MAX:
        return {
            "action": "block",
            "detector": "brute_force",
            "reason": f"{brute_count} attempts on '{target}' in {BRUTE_FORCE_WINDOW}s",
        }

    # 3. Bot check
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
# Routes — auth
# ---------------------------------------------------------------------------
@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    users = ref("users").get() or {}
    for uid, u in users.items():
        if u.get("email") == email:
            return jsonify({"error": "An account with this email already exists."}), 400

    user_id = str(uuid.uuid4())
    ref(f"users/{user_id}").set({
        "email": email,
        "password_hash": bcrypt.hash(password),
        "created_at": time.time(),
    })

    token = make_token()
    ref(f"sessions/{token}").set({"user_id": user_id, "created_at": time.time()})

    return jsonify({"token": token, "user_id": user_id})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    users = ref("users").get() or {}
    user_id, user = None, None
    for uid, u in users.items():
        if u.get("email") == email:
            user_id, user = uid, u
            break

    if not user or not bcrypt.verify(password, user["password_hash"]):
        return jsonify({"error": "Invalid email or password."}), 401

    token = make_token()
    ref(f"sessions/{token}").set({"user_id": user_id, "created_at": time.time()})

    return jsonify({"token": token, "user_id": user_id})


# ---------------------------------------------------------------------------
# Routes — API keys
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

    return jsonify({"key_id": key_id, "api_key": raw_key})


@app.route("/api/keys", methods=["GET"])
def list_keys():
    user_id = require_auth()
    if not user_id:
        return jsonify({"error": "Not authenticated."}), 401

    all_keys = ref("api_keys").get() or {}
    result = [
        {"key_id": kid, "label": k.get("label"), "active": k.get("active", True), "created_at": k.get("created_at")}
        for kid, k in all_keys.items()
        if k.get("user_id") == user_id
    ]
    return jsonify({"keys": result})


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
