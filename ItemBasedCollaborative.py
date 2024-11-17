import pandas as pd
import numpy as np

class ItemBasedRecommender:
    def __init__(self):
        # 유사도 행렬 불러오기
        self.similarity_matrix = pd.read_csv('data/item_similarity_matrix.csv', index_col=0)
    
    def recommend(self, input_news_ids, t):
        recommendation_scores = {}
        count_overlap = {}

        for news_id in input_news_ids:
            if news_id not in self.similarity_matrix.index:
                print(f"Warning: News ID {news_id} not found in similarity matrix.")
                continue

            # 각 기사에 대해 유사도 계산
            similar_items = self.similarity_matrix[news_id]
            
            for similar_news_id, similarity in similar_items.items():
                if similarity == 0 or similar_news_id in input_news_ids:
                    continue  # 유사도 0이거나 입력 기사 자체는 건너뛰기

                if similar_news_id not in recommendation_scores:
                    recommendation_scores[similar_news_id] = 0
                    count_overlap[similar_news_id] = 0

                # 특정 기사가 여러 input 기사와 유사도가 겹칠 경우 제일 높은 값과 겹친 횟수를 저장
                recommendation_scores[similar_news_id] = max(recommendation_scores[similar_news_id], similarity)
                count_overlap[similar_news_id] += 1

        # 겹친 횟수만큼 유사도 점수 값에서 0.1 만큼 보너스 점수를 더한다
        for news_id in recommendation_scores:
            n = count_overlap[news_id]
            recommendation_scores[news_id] += (n - 1) * 0.1

        # 상위 t개의 추천 기사 선택
        sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
        top_recommendations = sorted_recommendations[:t]

        # 유사도 점수에 따라 상위 t개의 (뉴스 기사 ID, 유사도 점수)의 튜플을 리스트 형식으로 반환
        return [(news_id, score) for news_id, score in top_recommendations]