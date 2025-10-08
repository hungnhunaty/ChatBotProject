"""train_phobert.py
Huấn luyện classifier dùng PhoBERT embeddings + sklearn LogisticRegression.

Cách chạy:
  python train_phobert.py
- Lần chạy đầu sẽ tải mô hình PhoBERT (khoảng 500MB).
- Nếu có GPU, script sẽ dùng GPU tự động để tăng tốc.
"""
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import joblib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def load_intents(path='intents.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_phobert(model_name='vinai/phobert-base'):
    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def embed_sentence(tokenizer, model, sentence, device='cpu'):
    # Tokenize sentence, run through model, perform mean pooling using attention mask
    inputs = tokenizer(sentence, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        attn = inputs['attention_mask'].unsqueeze(-1)  # shape (1, seq_len, 1)
        last = outputs.last_hidden_state  # shape (1, seq_len, hidden)
        summed = (last * attn).sum(1)    # sum over seq_len
        counts = attn.sum(1).clamp(min=1e-9)
        emb = (summed / counts).squeeze().cpu().numpy()
    return emb

def build_embeddings(intents, tokenizer, model, device='cpu'):
    sentences = []
    labels = []
    for it in intents['intents']:
        tag = it['tag']
        for p in it['patterns']:
            sentences.append(p)
            labels.append(tag)
    X = []
    for s in tqdm(sentences, desc='Embedding sentences'):
        X.append(embed_sentence(tokenizer, model, s, device=device))
    X = np.vstack(X)
    return X, labels, sentences

def main():
    intents = load_intents('intents.json')
    print('Loading PhoBERT (this may download the model the first time)...')
    tokenizer, model = get_phobert()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('Using GPU for embedding.')
        model.to(device)
    X, labels, sentences = build_embeddings(intents, tokenizer, model, device=device)
    # encode labels to integers for sklearn
    le = LabelEncoder()
    y = le.fit_transform(labels)
    clf = LogisticRegression(max_iter=2000)
    print('Training classifier on', X.shape[0], 'examples...')
    clf.fit(X, y)
    # save artifacts
    joblib.dump(clf, os.path.join(MODELS_DIR, 'classifier.joblib'))
    joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.joblib'))
    with open(os.path.join(MODELS_DIR, 'phobert_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'model_name': 'vinai/phobert-base'}, f, ensure_ascii=False, indent=2)
    print('Saved models to', MODELS_DIR)

if __name__ == '__main__':
    main()
