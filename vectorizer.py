# vectorizer.py

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import numpy as np

class Vectorizer:
    def __init__(self, news_df):
        self.news_df = news_df
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim_tfidf = None
        self.w2v_model = None
        self.cosine_sim_w2v = None
        self.sentence_model = None
        self.bert_vectors = None
        self.cosine_sim_bert = None

    def tfidf_vectorize(self, max_df=0.8, min_df=2, ngram_range=(1,2)):
        self.tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram_range)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.news_df['Content'])
        self.cosine_sim_tfidf = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print("TF-IDF 벡터화 완료.")

    def train_word2vec(self, vector_size=100, window=5, min_count=2, workers=4):
        tokens = self.news_df['Content'].apply(lambda x: x.split())
        self.w2v_model = Word2Vec(sentences=tokens, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        print("Word2Vec 모델 훈련 완료.")

    def compute_w2v_similarity(self):
        def get_article_vector(tokens):
            vectors = [self.w2v_model.wv[word] for word in tokens if word in self.w2v_model.wv]
            if vectors:
                return np.mean(vectors, axis=0)
            else:
                return np.zeros(self.w2v_model.vector_size)
        
        self.news_df['W2V_Vector'] = self.news_df['Content'].apply(lambda x: get_article_vector(x.split()))
        self.w2v_matrix = np.vstack(self.news_df['W2V_Vector'].values)
        self.cosine_sim_w2v = cosine_similarity(self.w2v_matrix, self.w2v_matrix)
        print("Word2Vec 유사도 계산 완료.")

    def compute_bert_embeddings(self, model_name='all-MiniLM-L6-v2', batch_size=64):
        self.sentence_model = SentenceTransformer(model_name)
        contents = self.news_df['Content'].tolist()
        self.bert_vectors = self.sentence_model.encode(contents, batch_size=batch_size, show_progress_bar=True)
        self.cosine_sim_bert = cosine_similarity(self.bert_vectors, self.bert_vectors)
        print("BERT 임베딩 및 유사도 계산 완료.")

    def compute_all(self):
        self.tfidf_vectorize()
        self.train_word2vec()
        self.compute_w2v_similarity()
        self.compute_bert_embeddings()
