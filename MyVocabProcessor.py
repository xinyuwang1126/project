import nltk
import numpy as np

class MyVocabProcessor(object):
    def __init__(self, vocabulary, tokenizer_fn=nltk.word_tokenize):
        self.vocab_len = len(vocabulary)
        self.vocabulary = {k: v for v, k in enumerate(vocabulary)}
        self.tokenizer_fn = tokenizer_fn

    def transform(self, X, max_doc_len):
        word_ids = np.ones([len(X), max_doc_len], np.int64) * (self.vocab_len - 1)      # padded by the word vector <unk>
        for n, row in enumerate(X):
            words = self.tokenizer_fn(row)
            for idx, word in enumerate(words):
                if idx >= max_doc_len:
                    break
                if word in self.vocabulary:
                    word_ids[n, idx] = self.vocabulary[word]
                else:
                    word_ids[n, idx] = np.random.choice(self.vocab_len)
        return word_ids
