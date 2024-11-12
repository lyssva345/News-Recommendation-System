# recommender.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

class Recommender:
    def __init__(self, news_df, behaviors_df, vectorizer):
        self.news_df = news_df
        self.behaviors_df = behaviors_df
        self.vectorizer = vectorizer
        self.indices = pd.Series(self.news_df.index, index=self.news_df['NewsID']).drop_duplicates()
        self.user_indices = pd.Series(self.behaviors_df.index, index=self.behaviors_df['UserID']).drop_duplicates()
        self.user_item_matrix = None
        self.user_sim = None
        self.prepare_collaborative_filtering()

    def prepare_collaborative_filtering(self):
        # 사용자-아이템 매트릭스 생성
        count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '))
        self.user_item_matrix = count_vectorizer.fit_transform(self.behaviors_df['History'])
        # 사용자 간 코사인 유사도 계산
        self.user_sim = cosine_similarity(self.user_item_matrix)
        print("협업 필터링 준비 완료.")

    def content_based_recommendations_with_scores(self, news_id, cosine_sim=None, top_n=10):
        if cosine_sim is None:
            cosine_sim = self.vectorizer.cosine_sim_tfidf
        try:
            idx = self.indices[news_id]
        except KeyError:
            print(f"뉴스 ID {news_id}가 데이터셋에 존재하지 않습니다.")
            return []
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]  # 자기 자신 제외
        news_indices = [i[0] for i in sim_scores]
        scores = [score for (_, score) in sim_scores]
        return list(zip(self.news_df['NewsID'].iloc[news_indices], scores))

    def collaborative_recommendations_with_scores(self, user_id, top_n=10):
        try:
            idx = self.user_indices[user_id]
        except KeyError:
            print(f"사용자 ID {user_id}가 데이터셋에 존재하지 않습니다.")
            return []
        sim_scores = list(enumerate(self.user_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]  # 자기 자신 제외
        similar_users = [i[0] for i in sim_scores]
        similar_users_histories = self.behaviors_df.iloc[similar_users]['History']
        all_articles = ' '.join(similar_users_histories).split(' ')
        article_counts = Counter(all_articles)
        user_read_articles = self.behaviors_df[self.behaviors_df['UserID'] == user_id]['History'].values[0].split(' ')
        recommended_articles = [article for article, count in article_counts.most_common() if article not in user_read_articles][:top_n]
        scores = [article_counts[article] for article in recommended_articles]
        return list(zip(recommended_articles, scores))

    def hybrid_recommendations(self, user_id, news_id, alpha=0.5, top_n=10):
        # 콘텐츠 기반 추천과 점수 가져오기
        content_recs = self.content_based_recommendations_with_scores(news_id, cosine_sim=self.vectorizer.cosine_sim_tfidf, top_n=top_n)
        # 협업 필터링 추천과 점수 가져오기
        collaborative_recs = self.collaborative_recommendations_with_scores(user_id, top_n=top_n)
        
        combined_scores = {}
        
        # 콘텐츠 기반 추천 점수에 가중치 적용
        for rec, score in content_recs:
            combined_scores[rec] = combined_scores.get(rec, 0) + alpha * score
        
        # 협업 필터링 추천 점수에 가중치 적용
        for rec, score in collaborative_recs:
            combined_scores[rec] = combined_scores.get(rec, 0) + (1 - alpha) * score
        
        # 최종 점수를 기준으로 정렬
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 top_n 개 추천 기사 추출
        top_recs = [rec for rec, score in sorted_recs[:top_n]]
        
        return top_recs


    def update_with_new_news(self, new_news_df):
        # 기존 뉴스 데이터에 새로운 뉴스 추가
        self.news_df = pd.concat([self.news_df, new_news_df], ignore_index=True)
        # 벡터화 및 유사도 재계산
        self.vectorizer.tfidf_vectorize()
        self.vectorizer.train_word2vec()
        self.vectorizer.compute_w2v_similarity()
        self.vectorizer.compute_bert_embeddings()
        # 업데이트된 유사도 매트릭스 재설정
        self.indices = pd.Series(self.news_df.index, index=self.news_df['NewsID']).drop_duplicates()
        print("새로운 뉴스 데이터가 추가되고, 벡터화 및 유사도 매트릭스가 업데이트되었습니다.")

    def update_with_new_user(self, new_behaviors_df):
        # 기존 사용자 행동 데이터에 새로운 사용자 추가
        self.behaviors_df = pd.concat([self.behaviors_df, new_behaviors_df], ignore_index=True)
        # 사용자-아이템 매트릭스 및 유사도 재계산
        self.prepare_collaborative_filtering()
        self.user_indices = pd.Series(self.behaviors_df.index, index=self.behaviors_df['UserID']).drop_duplicates()
        print("새로운 사용자 데이터가 추가되고, 협업 필터링 매트릭스가 업데이트되었습니다.")
