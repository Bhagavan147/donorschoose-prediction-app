import numpy as np
import re
import statistics
from wordfreq import zipf_frequency
import nltk
import joblib

nltk.download('stopwords')

stop_words = nltk.corpus.stopwords.words('english')

def text_clean(text):
  text = text.lower()
  text = re.sub(r'\\n|\\r|\\t|\\|\\a|\\b|\\f|\\v', ' ', text) # removing escape characters
  text = re.sub(r'[^a-z0-9 ]', ' ', text)
  text = re.sub(r'\s+', ' ', text).strip()
  text = [word for word in text.split() if word not in stop_words]
  return ' '.join(text)

def _syllables(word):
    w = re.sub(r'[^a-z]', '', word.lower())
    if not w: return 0
    groups = re.findall(r'[aeiou]+', w)
    cnt = len(groups)
    if w.endswith("e") and not w.endswith("le") and cnt > 1:
        cnt -= 1
    return max(1, cnt)

def readability_grade(title):
    text = (title or "").strip()
    text = re.sub(r'\\n|\\r|\\t|\\|\\a|\\b|\\f|\\v|\"', ' ', text) # removing escape characters
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    s_cnt = max(1, len(sentences))
    words = re.findall(r"[A-Za-z0-9']+", text)
    w_cnt = max(1, len(words))
    syll = sum(_syllables(w) for w in words)
    fkgl = 0.39 * (w_cnt / s_cnt) + 11.8 * (syll / w_cnt) - 15.59
    return round(fkgl, 3)

_REQUEST_PAT = re.compile(
    r"\b(help|need|looking for|looking to|we need|please|support|request|seeking|seek|donate|donation|want|would like|asking for|looking|requesting)\b",
    flags=re.I
)

def is_request(title):
    t = (title or "").lower()
    # direct starts-with-verb heuristic (e.g., "Help us...", "Need...")
    if re.match(r"^\s*(help|need|support|request|seeking|seek|looking|requesting)\b", t):
        return 1
    return int(bool(_REQUEST_PAT.search(t)))

def creativity_score(title):
    text = (title or "").strip().lower()
    words = re.findall(r"[a-z0-9']+", text)
    if not words:
        return 0.0
    # zipf_frequency ~ 1..7 (higher = common). invert to get "rarity"
    freqs = [zipf_frequency(w, "en") for w in words]
    mean_zipf = statistics.mean(freqs)
    rare_score = max(0.0, 7.0 - mean_zipf)   # larger => rarer vocabulary
    # small punctuation bonus for stylistic flair
    punct_bonus = 1.0 if re.search(r"[!,:;\"'()-]", title) else 0.0
    score = rare_score * 0.85 + punct_bonus * 0.15
    return round(score, 3)

def extract_title_features(features: dict) -> dict:
    project_title = features.get("project_title", "")

    title_features = {
        "title_readability_grade": 0.0,
        "is_title_request": 0,
        "title_creativity_score": 0.0,
        "cleaned_title_word_count": 0,
        "title_length": 0
    }

    if not project_title:
        return title_features
    
    title_features["title_readability_grade"] = np.log1p(readability_grade(project_title))
    title_features["is_title_request"] = is_request(project_title)
    title_features["title_creativity_score"] = np.log1p(creativity_score(project_title))
    title_features["cleaned_title_word_count"] = np.log1p(len(text_clean(project_title).split()))
    title_features["title_length"] = np.log1p(len(project_title))

    return title_features

sbert_model = joblib.load("models/sbert_model.pkl")

def title_embeddings(features: dict) -> dict:
    embeddings = sbert_model.encode(features.get("project_title"))
    return {f"title_emb_{i}": embeddings[i] for i in range(384)}