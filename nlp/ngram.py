"""Laplace-smoothed n-gram language model (Experiment 7)."""
import math
from collections import Counter, defaultdict

START, END = "<s>", "</s>"


class NGramModel:
    def __init__(self, n=2, k=1.0):
        self.n, self.k = n, k
        self.ngrams, self.contexts = Counter(), Counter()
        self.vocab = set()
        self.next_words = defaultdict(Counter)

    def pad(self, tokens):
        return [START] * (self.n - 1) + tokens + [END]

    def fit(self, sentences):
        for tokens in sentences:
            padded = self.pad(tokens)
            self.vocab.update(padded)
            for i in range(len(padded) - self.n + 1):
                gram = tuple(padded[i:i + self.n])
                self.ngrams[gram] += 1
                self.contexts[gram[:-1]] += 1
                self.next_words[gram[:-1]][gram[-1]] += 1
        return self

    def prob(self, context, word):
        context = tuple(context)
        return (self.ngrams[context + (word,)] + self.k) / (self.contexts[context] + self.k * len(self.vocab))

    def sequence_log_prob(self, tokens):
        padded = self.pad(tokens)
        return sum(math.log(self.prob(padded[i - self.n + 1:i], padded[i]))
                   for i in range(self.n - 1, len(padded)))

    def perplexity(self, sentences):
        log_sum, count = 0.0, 0
        for tokens in sentences:
            log_sum += self.sequence_log_prob(tokens)
            count += len(tokens) + 1
        return math.exp(-log_sum / count)

    def predict_next(self, context, top=3):
        context = tuple(context[-(self.n - 1):]) if self.n > 1 else ()
        if self.n > 1 and len(context) < self.n - 1:
            context = (START,) * (self.n - 1 - len(context)) + context
        scores = {w: self.prob(context, w) for w in self.vocab if w not in (START,)}
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top]
