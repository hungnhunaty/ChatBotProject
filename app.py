
"""
[\\\\\\app.py] - Flask server for Chatbot HUTECH (PhoBERT)
Routes:
  GET  /        -> returns UI page
  POST /api/chat -> receives JSON {message: "..."} and returns {reply: "..."}
  GET  /health   -> basic health check
"""
# -*- coding: utf-8 -*-

import json, random
from flask import Flask, render_template, request, jsonify
from model_utils import load_classifier, embed_sentence, models_exist

app = Flask(__name__)

# Load intents at startup for responses
with open('intents.json', 'r', encoding='utf-8') as f:
    INTENTS = json.load(f)

TOKENIZER, PHOBERT_MODEL, CLF, LABEL_ENCODER, DEVICE = None, None, None, None, None
if models_exist():
    try:
        TOKENIZER, PHOBERT_MODEL, CLF, LABEL_ENCODER, DEVICE = load_classifier()
        print('Loaded classifier and PhoBERT for inference.')
    except Exception as e:
        print('Error loading models:', e)
else:
    print('Models not found. Please run train_phobert.py to create models/.')

def get_response(user_text):
    # If models not loaded, instruct user to run training script
    if TOKENIZER is None:
        return 'Model chưa được huấn luyện. Vui lòng chạy python train_phobert.py trước.'
    emb = embed_sentence(TOKENIZER, PHOBERT_MODEL, user_text, device=DEVICE)
    y_pred = CLF.predict([emb])[0]
    tag = LABEL_ENCODER.inverse_transform([y_pred])[0]
    for it in INTENTS['intents']:
        if it['tag'] == tag:
            return random.choice(it.get('responses', ['Xin lỗi, mình chưa biết.']))
    return 'Xin lỗi, mình chưa có thông tin đó.'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json or {}
    text = data.get('message', '').strip()
    if not text:
        return jsonify({'reply': 'Vui lòng nhập câu hỏi.'})
    reply = get_response(text)
    return jsonify({'reply': reply})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'models_loaded': TOKENIZER is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
