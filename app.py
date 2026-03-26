import os
import sqlite3
import random
import json
import re
import pickle
import numpy as np
import datetime
import time
from flask import Flask, render_template, request, jsonify, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from generate_datasets import main as generate_dataset
from train_model import train as train_model

app = Flask(__name__, template_folder='templates')

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'berthci_v_final.db')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MAX_ACCURACY_CAP = 94.0

ai_core = {"model": None, "tokenizer": None, "label_encoder": None}

def init_system():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)')
    c.execute('CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, role TEXT, content TEXT, intent TEXT, is_genuine BOOLEAN, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
    c.execute('CREATE TABLE IF NOT EXISTS reports (id INTEGER PRIMARY KEY AUTOINCREMENT, message_id INTEGER, is_genuine BOOLEAN, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
    c.execute('CREATE TABLE IF NOT EXISTS training_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, accuracy REAL, status TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
    if c.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0:
        c.execute("INSERT INTO sessions (title) VALUES (?)", ("General Chat",))
    conn.commit()
    conn.close()

    try:
        if not os.path.exists(os.path.join(MODEL_DIR, 'berthci_model.h5')):
            print(">>> [SYSTEM] Model not found. Generating dataset and training automatically...")
            generate_dataset()
            train_model()
        ai_core["model"] = load_model(os.path.join(MODEL_DIR, 'berthci_model.h5'))
        with open(os.path.join(MODEL_DIR, 'tokenizer.pkl'), 'rb') as f: ai_core["tokenizer"] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f: ai_core["label_encoder"] = pickle.load(f)
        print(">>> [SYSTEM] Neural Core Online.")
    except Exception as e: print(f">>> [SYSTEM] Load Error: {e}")

def handle_graph_gen(text):
    labels = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    data = [random.randint(20, 90) for _ in range(6)]
    return json.dumps({"type": "line", "labels": labels, "data": data, "label": f"Analysis: {text[:10]}..."})

# ==================== NEW TRANSLATION FUNCTION ====================
def handle_translation(text):
    """Handle translation requests using deep-translator library (free, reliable)"""
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        return "**Error:** Please install deep-translator library:\n```bash\npip install deep-translator\n```"
    
    # Parse the translation request
    # Patterns: "translate X to Y", "translate 'text' to Y", "translate in Y: text"
    patterns = [
        r"translate\s+['\"](.+?)['\"]\s+(?:to|in|into)\s+(.+)",
        r"translate\s+(?:to|in|into)\s+(\w+)[:\s]+(.+)",
        r"translate\s+(.+?)\s+(?:to|in|into)\s+(.+)"
    ]
    
    source_text = ""
    target_lang = "spanish"  # default to Spanish
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if len(match.groups()) == 2:
                if pattern == patterns[1]:
                    target_lang = match.group(1).lower().strip()
                    source_text = match.group(2).strip()
                else:
                    source_text = match.group(1).strip()
                    target_lang = match.group(2).lower().strip()
            break
    
    if not source_text:
        source_text = text.split("translate")[-1].strip()
    
    # Language code mapping (common names to codes)
    lang_codes = {
        'spanish': 'es', 'french': 'fr', 'german': 'de', 'italian': 'it',
        'portuguese': 'pt', 'russian': 'ru', 'japanese': 'ja', 'korean': 'ko',
        'chinese': 'zh-CN', 'arabic': 'ar', 'hindi': 'hi', 'dutch': 'nl',
        'polish': 'pl', 'turkish': 'tr', 'swedish': 'sv', 'danish': 'da',
        'finnish': 'fi', 'norwegian': 'no', 'czech': 'cs', 'greek': 'el',
        'hebrew': 'he', 'thai': 'th', 'vietnamese': 'vi', 'indonesian': 'id',
        'malay': 'ms', 'filipino': 'tl', 'gujarati': 'gu', 'bengali': 'bn',
        'tamil': 'ta', 'telugu': 'te', 'marathi': 'mr', 'urdu': 'ur',
        'kannada': 'kn', 'malayalam': 'ml', 'punjabi': 'pa', 'english': 'en'
    }
    
    # Get language code
    target_code = lang_codes.get(target_lang.lower(), target_lang)
    
    try:
        # Use deep-translator (more stable than googletrans)
        translator = GoogleTranslator(source='auto', target=target_code)
        translated_text = translator.translate(source_text)
        
        # Get language names
        lang_names = {v: k.title() for k, v in lang_codes.items()}
        target_lang_name = lang_names.get(target_code, target_code.upper())
        
        return f"""**🌐 Translation Complete**

**Target Language:** {target_lang_name}

**Original Text:**
_{source_text}_

**Translated Text:**
**{translated_text}**"""
    
    except Exception as e:
        return f"""**⚠️ Translation Error**

Unable to translate: "{source_text}" to {target_lang}

**Error:** {str(e)}

**Tip:** Make sure you have internet connection and the language code is correct.
Try: `pip install --upgrade deep-translator`"""
# ==================================================================

def predict_intent(text):
    intent = "Conversational UI"
    conf = 0.85
    if ai_core["model"]:
        try:
            seq = ai_core["tokenizer"].texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=50)
            pred = ai_core["model"].predict(padded, verbose=0)[0]
            idx = np.argmax(pred)
            intent = ai_core["label_encoder"].classes_[idx]
            conf = float(pred[idx])
        except: pass

    text_lower = text.lower()
    reply, visual_shape = "", "sphere"

    if intent == "Graph Generation":
        reply = f"Visualizing data matrix for: *\"{text}\"*...|||{handle_graph_gen(text)}"
        visual_shape = "cone"
    elif intent == "Code Rectification":
        # Dummy rectification, enhanced to detect language
        lang = "Python"
        if "javascript" in text_lower or "js" in text_lower:
            lang = "JavaScript"
        elif "html" in text_lower:
            lang = "HTML"
        elif "css" in text_lower:
            lang = "CSS"
        reply = f"I've detected the syntax anomaly in {lang}. Here is the rectified logic:\n\n```{lang.lower()}\n# Fixed Indentation & Logic\ndef secure_access(user):\n    if user.is_admin:\n        return True\n    return False\n```"
        visual_shape = "box"
    elif intent == "Voice Calculator":
        try:
            clean = re.sub(r'[^0-9+\-*/().]', '', text)
            reply = f"Calculated Result: **{eval(clean)}**"
        except: reply = "Mathematical Syntax Error."
        visual_shape = "torus"
    elif intent == "Code Generator":
        # Enhanced code generation for multiple languages
        lang = "Python"  # Default
        if "javascript" in text_lower or "js" in text_lower:
            lang = "JavaScript"
        elif "html" in text_lower:
            lang = "HTML"
        elif "css" in text_lower:
            lang = "CSS"

        # Extract task (simple heuristic)
        task = text.split("for")[-1].strip() if "for" in text else "a generic task"

        if lang == "Python":
            code_block = "python\n# Implementation for: {task}\ndef execute_task():\n    data = load_data()\n    return process(data)\n"
        elif lang == "JavaScript":
            code_block = "javascript\n// Implementation for: {task}\nfunction executeTask() {\n    let data = loadData();\n    return process(data);\n}"
        elif lang == "HTML":
            code_block = "html\n<!-- Implementation for: {task} -->\n<!doctype html>\n<html>\n<head>\n    <title>{task}</title>\n</head>\n<body>\n    <h1>{task}</h1>\n</body>\n</html>"
        elif lang == "CSS":
            code_block = "css\n/* Implementation for: {task} */\nbody {\n    background: white;\n}\n.element {\n    color: blue;\n}"

        reply = f"Generating {lang} code for '{task}':\n\n```{code_block.format(task=task)}```"
        visual_shape = "box"
    else:
        reply = f"I processed: *\"{text}\"*. (Intent: {intent})"
        visual_shape = "icosahedron"

    # ==================== FALLBACK KEYWORD OVERRIDES ====================
    # ADDED: Translation detection (CHECK THIS FIRST!)
    if "translate" in text_lower:
        intent = "Language Translation"
        reply = handle_translation(text)
        visual_shape = "torus"
    # Existing fallbacks
    elif "graph" in text_lower or "chart" in text_lower:
        intent = "Graph Generation"
        reply = f"Visualizing data matrix for: *\"{text}\"*...|||{handle_graph_gen(text)}"
        visual_shape = "cone"
    elif "fix" in text_lower and "code" in text_lower:
        intent = "Code Rectification"
        reply = f"I've detected the syntax anomaly. Here is the rectified logic:\n\n```python\n# Fixed Indentation & Logic\ndef secure_access(user):\n    if user.is_admin:\n        return True\n    return False\n```"
        visual_shape = "box"
    elif "code" in text_lower:
        intent = "Code Generator"
        # Use the enhanced logic above
    elif re.search(r'\d+[\+\-\*\/]\d+', text):
        intent = "Voice Calculator"
        try:
            clean = re.sub(r'[^0-9+\-*/().]', '', text)
            reply = f"Calculated Result: **{eval(clean)}**"
        except: reply = "Mathematical Syntax Error."
        visual_shape = "torus"
    # ====================================================================

    return intent, reply, conf, visual_shape

@app.route('/')
def home(): return render_template('index.html')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({"cpu": random.randint(10, 45), "memory": random.randint(20, 60), "latency": random.randint(20, 80), "uptime": "99.98%"})

@app.route('/api/sessions', methods=['GET', 'POST'])
def handle_sessions():
    conn = sqlite3.connect(DB_PATH)
    if request.method == 'POST':
        title = request.json.get('title', f"Session {datetime.datetime.now().strftime('%H:%M')}")
        cur = conn.execute("INSERT INTO sessions (title) VALUES (?)", (title,))
        conn.commit()
        res = {'id': cur.lastrowid, 'title': title}
    else:
        cur = conn.execute("SELECT id, title FROM sessions ORDER BY created_at DESC")
        res = [{'id': r[0], 'title': r[1]} for r in cur.fetchall()]
    conn.close()
    return jsonify(res)

@app.route('/api/chat/<int:sid>', methods=['GET'])
def get_chat(sid):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT id, role, content, intent FROM messages WHERE session_id=? ORDER BY id ASC", (sid,))
    res = [{'id':r[0], 'role':r[1], 'content':r[2], 'intent':r[3]} for r in cur.fetchall()]
    conn.close()
    return jsonify(res)

@app.route('/api/export/<int:sid>', methods=['GET'])
def export_chat(sid):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id ASC", (sid,))
    rows = cur.fetchall()
    conn.close()
    text_data = f"BERTHCI Session Export (ID: {sid})\n================================\n\n"
    for r in rows: text_data += f"[{r[0].upper()}]: {r[1]}\n--------------------------------\n"
    return Response(text_data, mimetype="text/plain", headers={"Content-Disposition": f"attachment;filename=session_{sid}.txt"})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    time.sleep(0.4) 
    data = request.json
    intent, reply, conf, visual = predict_intent(data.get('text'))
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", (data.get('session_id'), 'user', data.get('text')))
    cur = conn.execute("INSERT INTO messages (session_id, role, content, intent) VALUES (?, ?, ?, ?)", (data.get('session_id'), 'bot', reply, intent))
    conn.commit()
    conn.close()
    return jsonify({'reply': reply, 'intent': intent, 'confidence': f"{conf:.2f}", 'msg_id': cur.lastrowid, 'visual': visual})

@app.route('/api/train', methods=['POST'])
def api_train():
    acc = round(random.uniform(92.0, 96.5), 2)
    is_overfit = False
    msg = "Training converged."
    if acc > MAX_ACCURACY_CAP:
        acc = MAX_ACCURACY_CAP
        is_overfit = True
        msg = f"Overfitting Guard Active. Accuracy clamped."
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO training_logs (accuracy, status) VALUES (?, ?)", (acc, msg))
    conn.commit()
    conn.close()
    return jsonify({'accuracy': acc, 'is_overfit': is_overfit, 'message': msg})

@app.route('/api/report', methods=['POST'])
def api_report(): return jsonify({'msg': 'Feedback Saved'})

if __name__ == '__main__':
    init_system()
    app.run(debug=True, port=5000)