from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import pandas as pd


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = 'yelp_academic_dataset_review.json'
        for review in pd.read_json(corpus_path, chunksize=10000, lines=True):
            text_review = review['text']
            for rev in text_review:
                lines = rev.lower().replace('!', '.').replace('?', '.').split('.')
                # assume there's one document per line, tokens separated by whitespace
                for line in lines:
                    yield utils.simple_preprocess(line)


sentences = MyCorpus()
model = gensim.models.Word2Vec(min_count=10, window=5, vector_size=100, workers=2, sentences=sentences)
model.save("word2vec.model")
