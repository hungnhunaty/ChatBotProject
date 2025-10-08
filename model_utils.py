"""model_utils.py
- load_classifier(): tải classifier + label encoder + tokenizer + phobert model cho inference.
- embed_sentence(): chuyển câu thành vector embedding (mean pooling).
"""
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def models_exist():
    return (os.path.exists(os.path.join(MODELS_DIR, 'classifier.joblib')) and
            os.path.exists(os.path.join(MODELS_DIR, 'label_encoder.joblib')) and
            os.path.exists(os.path.join(MODELS_DIR, 'phobert_meta.json')))

def load_classifier():
    if not models_exist():
        raise FileNotFoundError('Models not found. Run train_phobert.py first.')
    clf = joblib.load(os.path.join(MODELS_DIR, 'classifier.joblib'))
    le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.joblib'))
    with open(os.path.join(MODELS_DIR, 'phobert_meta.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(meta['model_name'], use_fast=False)
    model = AutoModel.from_pretrained(meta['model_name'])
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        model.to(device)
    return tokenizer, model, clf, le, device

def embed_sentence(tokenizer, model, sentence, device='cpu'):
    inputs = tokenizer(sentence, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        attn = inputs['attention_mask'].unsqueeze(-1)
        last = outputs.last_hidden_state
        summed = (last * attn).sum(1)
        counts = attn.sum(1).clamp(min=1e-9)
        emb = (summed / counts).squeeze().cpu().numpy()
    return emb
