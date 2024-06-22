import nltk
import gensim
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from tensorflow.keras.preprocessing.sequence import pad_sequences
import yake
import pickle
import numpy as np
import sys
from src.nlpyoutube.logger import logging
from src.nlpyoutube.exception import CustomException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DataTransformation:
    def __init__(self):
        self.stop = set(stopwords.words('english'))
        self.lemma = WordNetLemmatizer()
        self.tokenizer = None
        self.maxlen = 150
        self.yake_kw = yake.KeywordExtractor(n=1, top=20)

    def load_tokenizer(self, filepath):
        try:
            with open(filepath, 'rb') as file:
                self.tokenizer = pickle.load(file)
        except Exception as e:
            raise CustomException(e,sys)

    def preprocess(self, text):
        tokens = gensim.utils.simple_preprocess(str(text), deacc=True)
        stop_free = [word for word in tokens if word not in self.stop]
        pos_tags = pos_tag(stop_free)
        
        normalized = []
        for word, tag in pos_tags:
            simplified_tag = self.simplify(tag)
            if simplified_tag == 'proper_noun':
                normalized.append(word)
            elif simplified_tag:
                normalized.append(self.lemma.lemmatize(word, pos=simplified_tag))
        
        UniqW = Counter(normalized)
        normalized = UniqW.keys()
        return ' '.join(normalized)

    def simplify(self, penn_tag):
        tag_prefix = penn_tag[0]
        if penn_tag in ['NNP', 'NNPS']:
            return 'proper_noun'
        elif tag_prefix == 'N':
            return wordnet.NOUN
        else:
            return wordnet.ADJ

    def extract_keywords(self, text):
        keywords = self.yake_kw.extract_keywords(text)
        return [kw for kw, _ in keywords]

    def preprocess_text(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=self.maxlen)
        return padded_sequence

    def calculate_cosine_similarity(self, texts):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return cosine_sim_matrix