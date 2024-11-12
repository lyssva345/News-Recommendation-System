# main.py

from data_loader import DataLoader
from vectorizer import Vectorizer
from recommender import Recommender

def main():
    # 데이터 경로 설정
    news_path = '/Users/yunseo/Desktop/ml/data/training/news.tsv'
    behaviors_path = '/Users/yunseo/Desktop/ml/data/training/behaviors.tsv'
    
    # # 데이터 로딩 및 전처리
    data_loader = DataLoader(news_path, behaviors_path)
    news_df, behaviors_df = data_loader.get_data()
    # 데이터 로딩 및 전처리 (샘플 크기 지정, 예: 1000)
    # data_loader = DataLoader(news_path, behaviors_path, sample_size=1000)
    # news_df, behaviors_df = data_loader.get_data()
    
    # 벡터화 및 유사도 계산
    vectorizer = Vectorizer(news_df)
    vectorizer.compute_all()
    
    # 추천 시스템 초기화
    recommender = Recommender(news_df, behaviors_df, vectorizer)
    
    # 추천 예시
    sample_news_id = news_df['NewsID'].iloc[0]
    sample_user_id = behaviors_df['UserID'].iloc[0]
    
    # 콘텐츠 기반 추천
    content_recs = recommender.content_based_recommendations_with_scores(sample_news_id)
    print("\n추천된 기사 ID (콘텐츠 기반):")
    print(content_recs)
    
    # 협업 필터링 추천
    collaborative_recs = recommender.collaborative_recommendations_with_scores(sample_user_id)
    print("\n추천된 기사 ID (협업 필터링):")
    print(collaborative_recs)
    
    # 하이브리드 추천
    hybrid_recs = recommender.hybrid_recommendations(sample_user_id, sample_news_id, alpha=0.6, top_n=10)
    print("\n추천된 기사 ID (하이브리드):")
    print(hybrid_recs)
    
    # 새로운 뉴스 데이터 추가 예시
    # new_news_df = pd.DataFrame([...])  # 새로운 뉴스 데이터 프레임 생성
    # recommender.update_with_new_news(new_news_df)
    
    # 새로운 사용자 데이터 추가 예시
    # new_behaviors_df = pd.DataFrame([...])  # 새로운 사용자 행동 데이터 프레임 생성
    # recommender.update_with_new_user(new_behaviors_df)

if __name__ == "__main__":
    main()
