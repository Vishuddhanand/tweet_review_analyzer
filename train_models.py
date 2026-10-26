"""Train every model used by the web app and save it as a .pkl file in models/.

    python train_models.py                      # uses data/tweets_sample.csv
    python train_models.py path/to/dataset.csv  # Twitter US Airline / Sentiment140 / (text,label)
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from nlp.ngram import NGramModel
from nlp.preprocessing import LEMMATIZER, load_tweets, preprocess, tokenize_plain

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
N_TOPICS, N_LSI = 5, 8


def save(obj, name):
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(obj, path, compress=3)
    print(f"  saved models/{name:28s} {os.path.getsize(path) / 1024:8.1f} KB")


def main(path):
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_tweets(path)
    use_lemma = LEMMATIZER is not None
    df["processed"] = df["clean"].apply(lambda t: preprocess(t, use_lemma))
    labels = sorted(df["label"].unique())
    print(f"Loaded {len(df)} tweets from {path}; classes={df['label'].value_counts().to_dict()}")
    print("Lemmatization:", "WordNet" if use_lemma else "off (NLTK wordnet data not installed)")

    # --- TF-IDF (keywords, similar-review search, topic model input) -------------
    tfidf = TfidfVectorizer(sublinear_tf=True)
    X = tfidf.fit_transform(df["processed"])
    terms = np.array(tfidf.get_feature_names_out())

    # --- NMF topics ----------------------------------------------------------------
    nmf = NMF(n_components=N_TOPICS, init="nndsvda", random_state=42, max_iter=500)
    df["topic"] = nmf.fit_transform(X).argmax(axis=1)
    topic_terms = [[terms[i] for i in np.argsort(c)[::-1][:5]] for c in nmf.components_]

    # --- LSI (TruncatedSVD) --------------------------------------------------------
    lsi_vec = TfidfVectorizer(stop_words="english")
    lsi_X = lsi_vec.fit_transform(df["clean"])
    n_lsi = max(2, min(N_LSI, lsi_X.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_lsi, random_state=42)
    norm = Normalizer()
    lsi_docs = norm.fit_transform(svd.fit_transform(lsi_X))

    # --- Classifiers (TF-IDF 1-2grams + LR / NB / SVM) -------------------------------
    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(df["processed"], df["label"], test_size=0.25,
                                              stratify=stratify, random_state=42)
    makers = {
        "Logistic Regression": lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                                     LogisticRegression(max_iter=2000)),
        "Multinomial NB": lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                                MultinomialNB(alpha=0.5)),
        "Linear SVM": lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                            LinearSVC()),
    }
    metrics = []
    confusion = {}
    classifiers = {}
    for name, make in makers.items():
        pred = make().fit(X_tr, y_tr).predict(X_te)
        p, r, f, _ = precision_recall_fscore_support(y_te, pred, labels=labels, average="macro", zero_division=0)
        metrics.append({"model": name, "accuracy": round(accuracy_score(y_te, pred), 3),
                        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)})
        confusion[name] = confusion_matrix(y_te, pred, labels=labels).tolist()
        classifiers[name] = make().fit(df["processed"], df["label"])  # final model: all data
        print(f"  {name:20s} test acc={metrics[-1]['accuracy']:.3f} macro-F1={metrics[-1]['f1']:.3f}")

    # --- N-gram language models ------------------------------------------------------
    sentences = [tokenize_plain(t) for t in df["clean"]]
    tr, te = train_test_split(sentences, test_size=0.2, random_state=42)
    ngrams, ppl = {}, {}
    for n in (1, 2, 3):
        ngrams[n] = NGramModel(n=n, k=1.0).fit(sentences)
        ppl[n] = round(NGramModel(n=n, k=1.0).fit(tr).perplexity(te), 2)

    # --- Word2Vec (optional, needs gensim) ---------------------------------------------
    w2v = None
    try:
        from gensim.models import Word2Vec
        w2v = Word2Vec(sentences=[t.split() for t in df["processed"]], vector_size=50, window=5,
                       min_count=1, workers=1, epochs=100, seed=42)
    except Exception as e:  # pragma: no cover
        print("  Word2Vec skipped:", e)

    # --- Save -------------------------------------------------------------------------
    print("Saving models:")
    save(tfidf, "tfidf_vectorizer.pkl")
    save(nmf, "nmf_topics.pkl")
    save({"vectorizer": lsi_vec, "svd": svd, "normalizer": norm, "doc_vectors": lsi_docs}, "lsi_model.pkl")
    for name, model in classifiers.items():
        save(model, "clf_" + name.lower().replace(" ", "_") + ".pkl")
    save(ngrams, "ngram_models.pkl")
    if w2v is not None:
        save(w2v, "word2vec.pkl")

    mean_scores = np.asarray(X.mean(axis=0)).ravel()
    top = np.argsort(mean_scores)[::-1][:8]
    meta = {
        "lemmatize": use_lemma,
        "labels": labels,
        "reviews": len(df),
        "vocab": int(len(terms)),
        "avg_words": round(float(df["clean"].str.split().str.len().mean()), 1),
        "classes": df["label"].value_counts().sort_index().to_dict(),
        "keywords": [[terms[i], round(float(mean_scores[i] / mean_scores[top[0]] * 100))] for i in top],
        "topics": [[k, int((df["topic"] == k).sum()), ", ".join(topic_terms[k])] for k in range(N_TOPICS)],
        "metrics": metrics, "confusion": confusion, "perplexity": ppl,
        "test_size": int(len(y_te)),
        "lsi_explained_variance": round(float(svd.explained_variance_ratio_.sum()), 3),
        "trained_at": time.strftime("%Y-%m-%d %H:%M"), "source": os.path.basename(path),
    }
    save({"meta": meta,
          "corpus": df[["text", "label", "clean", "processed", "topic"]].reset_index(drop=True),
          "tfidf_matrix": X}, "corpus_and_meta.pkl")
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "perplexity": ppl}, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "tweets_sample.csv"))
