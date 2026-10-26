# Twitter Tweet Review Analysis (NLP mini project)

Flask web app that loads **pre-trained `.pkl` models** to analyse tweet reviews:
cleaned text + tokens, review-class prediction (3 classifiers), NMF topic, TF-IDF key terms,
similar reviews (cosine similarity) and bigram next-word suggestions.

## Run
    pip install -r requirements.txt
    python app.py            # open http://127.0.0.1:5000

The trained models are already in `models/`. To retrain (e.g. on the full Twitter US Airline
Sentiment or Sentiment140 CSV) and regenerate the pkl files:

    python train_models.py                       # data/tweets_sample.csv (48 demo tweets)
    python train_models.py path/to/dataset.csv

For WordNet lemmatization run once: `python -c "import nltk; nltk.download('wordnet')"`
before training (the choice is stored with the models, so the app stays consistent).

## Model files (`models/`)
| File | Model | Used for |
|---|---|---|
| tfidf_vectorizer.pkl | TfidfVectorizer | keywords, similar-review search |
| nmf_topics.pkl | NMF (5 topics) | closest topic |
| lsi_model.pkl | TF-IDF + TruncatedSVD + Normalizer + doc vectors | LSI semantic space |
| clf_logistic_regression.pkl / clf_multinomial_nb.pkl / clf_linear_svm.pkl | TF-IDF(1-2gram) pipelines | review-class prediction |
| ngram_models.pkl | Unigram/bigram/trigram (Laplace) | next word, perplexity |
| word2vec.pkl | gensim Word2Vec (50-d) | embeddings |
| corpus_and_meta.pkl | corpus, TF-IDF matrix, dataset profile, metrics | dashboard |

## API
`POST /api/analyze` with JSON `{"tweet": "...", "model": "Logistic Regression"}` returns the analysis as JSON.

> The bundled dataset has only 48 demo tweets, so probabilities/metrics show that the app works, not model quality.

## Layout
    app.py  train_models.py  nlp/{preprocessing,ngram}.py  templates/  static/  data/  models/
