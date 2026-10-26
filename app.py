"""Tweet Review Analyzer - Flask web app that loads the trained .pkl models."""
import os
import re

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

from nlp.ngram import NGramModel  # noqa: F401  (needed to unpickle ngram_models.pkl)
from nlp.preprocessing import clean_tweet, preprocess, tokenize_plain

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")

if not os.path.exists(os.path.join(MODEL_DIR, "corpus_and_meta.pkl")):
    raise SystemExit("Model files not found. Run:  python train_models.py")

# ---- load the pickled models once at start-up ---------------------------------------
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
nmf = joblib.load(os.path.join(MODEL_DIR, "nmf_topics.pkl"))
lsi = joblib.load(os.path.join(MODEL_DIR, "lsi_model.pkl"))
ngrams = joblib.load(os.path.join(MODEL_DIR, "ngram_models.pkl"))
bundle = joblib.load(os.path.join(MODEL_DIR, "corpus_and_meta.pkl"))
CLASSIFIERS = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "clf_logistic_regression.pkl")),
    "Multinomial NB": joblib.load(os.path.join(MODEL_DIR, "clf_multinomial_nb.pkl")),
    "Linear SVM": joblib.load(os.path.join(MODEL_DIR, "clf_linear_svm.pkl")),
}
META, CORPUS, X = bundle["meta"], bundle["corpus"], bundle["tfidf_matrix"]
TERMS = np.array(tfidf.get_feature_names_out())
TOPIC_TERMS = {k: t for k, _, t in META["topics"]}
DEFAULT_MODEL = "Logistic Regression"

app = Flask(__name__)


def class_probabilities(model, processed):
    """Class probabilities; LinearSVC has no predict_proba so use a softmax of its margins."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([processed])[0]
    else:
        d = np.atleast_1d(model.decision_function([processed])[0])
        e = np.exp(d - d.max())
        probs = e / e.sum()
    classes = model.classes_
    return sorted(((str(c), round(float(p) * 100, 1)) for c, p in zip(classes, probs)), key=lambda x: -x[1])


def analyze(text, model_name=DEFAULT_MODEL):
    clean = clean_tweet(text)
    processed = preprocess(clean, META["lemmatize"])
    v = tfidf.transform([processed])
    scores = v.toarray().ravel()
    sims = cosine_similarity(v, X).ravel()
    topic = int(nmf.transform(v).argmax())
    toks = tokenize_plain(clean)
    bigram = ngrams[2]
    return {
        "clean": clean,
        "tokens": processed.split(),
        "hashtags": re.findall(r"#\w+", text),
        "mentions": re.findall(r"@\w+", text),
        "model": model_name,
        "probs": class_probabilities(CLASSIFIERS[model_name], processed),
        "topic": topic,
        "topic_terms": TOPIC_TERMS[topic],
        "keyterms": [(TERMS[i], round(float(scores[i]), 3)) for i in np.argsort(scores)[::-1][:6] if scores[i] > 0],
        "similar": [(round(float(sims[j]), 3), CORPUS["label"][j], CORPUS["text"][j])
                    for j in np.argsort(sims)[::-1][:3] if sims[j] > 0],
        # only suggest when the last word was seen in training (otherwise it is a uniform guess)
        "next_words": [(w, round(p, 4)) for w, p in bigram.predict_next(toks[-1:], top=4) if w.isalpha()][:3]
                      if toks and bigram.contexts[(toks[-1],)] > 0 else [],
        "perplexity": round(ngrams[2].perplexity([toks]), 1) if toks else None,
    }


@app.route("/")
def index():
    tweet = request.args.get("tweet", "").strip()
    model_name = request.args.get("model", DEFAULT_MODEL)
    if model_name not in CLASSIFIERS:
        model_name = DEFAULT_MODEL
    result = analyze(tweet, model_name) if tweet else None
    p = {**META, "max_class": max(META["classes"].values())}
    return render_template("index.html", p=p, tweet=tweet, r=result, model_name=model_name,
                           models=list(CLASSIFIERS))


@app.route("/api/analyze", methods=["POST", "GET"])
def api_analyze():
    data = request.get_json(silent=True) or request.values
    text = (data.get("tweet") or "").strip()
    model_name = data.get("model") or DEFAULT_MODEL
    if not text:
        return jsonify(error="Provide a 'tweet' field."), 400
    if model_name not in CLASSIFIERS:
        return jsonify(error=f"Unknown model. Choose one of {list(CLASSIFIERS)}"), 400
    return jsonify(analyze(text, model_name))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
