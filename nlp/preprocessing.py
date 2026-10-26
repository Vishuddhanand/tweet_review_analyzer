"""Shared text cleaning / loading (from utils.py and exp9 of the lab code)."""
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
RT_RE = re.compile(r"^rt\s+", re.I)
ELONG_RE = re.compile(r"(.)\1{2,}")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
STOP = set(ENGLISH_STOP_WORDS) - {"not", "no", "never", "nor"}  # keep negations


def _make_lemmatizer():
    """WordNet lemmatizer if the NLTK data is installed, otherwise None."""
    try:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        lem.lemmatize("tests")  # raises LookupError if wordnet is missing
        return lem
    except Exception:
        return None


LEMMATIZER = _make_lemmatizer()


def clean_tweet(text):
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = RT_RE.sub("", text)
    text = text.replace("#", "")
    text = ELONG_RE.sub(r"\1\1", text)
    return SPACE_RE.sub(" ", text).strip().lower()


def preprocess(text, lemmatize=None):
    """Tokenize, drop stop-words (keep negations) and optionally lemmatize.

    `lemmatize=None` means "use WordNet if available". The trained models record
    which choice was used so the web app always preprocesses the same way.
    """
    if lemmatize is None:
        lemmatize = LEMMATIZER is not None
    tokens = [t for t in TOKEN_RE.findall(text) if t not in STOP and len(t) > 1]
    if lemmatize and LEMMATIZER is not None:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def tokenize_plain(text):
    """Simple tokenizer used by the n-gram model."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def load_tweets(path):
    """Load a tweet CSV: sample (text,label), US-Airline, or Sentiment140 format."""
    with open(path, encoding="latin-1") as f:
        first = f.readline().lower()
    if "airline_sentiment" in first:
        df = pd.read_csv(path, encoding="utf-8")
        df["label"] = df["airline_sentiment"]
    elif first.startswith("text"):
        df = pd.read_csv(path, encoding="utf-8")
    else:
        names = ["target", "id", "date", "flag", "user", "text"]
        df = pd.read_csv(path, header=None, names=names, encoding="latin-1")
        df["label"] = df["target"].map({0: "negative", 2: "neutral", 4: "positive"})
        df = df[["text", "label"]]
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["clean"] = df["text"].apply(clean_tweet)
    return df
