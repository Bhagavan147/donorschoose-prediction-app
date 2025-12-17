import numpy as np
import re
import nltk
import html
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
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
    return round(fkgl, 2)

def basic_clean(text):
  if not text:
      return ""
  text = html.unescape(str(text))    # fix &amp; etc.
  text = re.sub(r'(?:\\r\\n|\\r|\\n)+', ' ', text)
  text = re.sub(r'\s+', ' ', text)
  return text.strip()

analyzer = SentimentIntensityAnalyzer()

def sentiment_vader(text):
  text = basic_clean(text)
  return round(analyzer.polarity_scores(text)['compound'], 3)

def sentiment_subjectivity(text):
  text = basic_clean(text)
  return round(TextBlob(text).sentiment.subjectivity, 3)


def extract_essay_features(features: dict) -> dict:
    project_essay = features.get("project_essay", "")
    essay_features = {
        "essay_readability_grade": 0.0,
        "cleaned_essay_word_count": 0,
        "essay_length": 0,
        "essay_sentence_count": 0,
        "essay_paragraph_count": 0,
        "essay_sentiment": 0.0,
        "essay_subjectivity": 0.0
    }

    if not project_essay:
        return essay_features

    essay_features["essay_readability_grade"] = np.log1p(readability_grade(project_essay))
    essay_features["cleaned_essay_word_count"] = np.log1p(len(text_clean(project_essay).split()))
    essay_features["essay_length"] = np.log1p(len(project_essay))
    essay_features["essay_sentence_count"] = np.log1p(len(re.split(r'[.!?]+\s*', project_essay)))
    essay_features["essay_paragraph_count"] = np.log1p(len([p for p in re.split(r'(?:\\r\\n|\\r|\\n)+', project_essay) if p.strip()]))
    essay_features["essay_sentiment"] = np.log1p(sentiment_vader(project_essay))
    essay_features["essay_subjectivity"] = np.log1p(sentiment_subjectivity(project_essay))

    return essay_features

sbert_model = joblib.load("models/sbert_model.pkl")

def essay_embeddings(features: dict) -> dict:
    embeddings = sbert_model.encode(features.get("project_essay"))
    return {f"essay_emb_{i}": embeddings[i] for i in range(384)}