import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 파일 경로 설정
news_path = 'news_select.tsv'

# 뉴스 데이터 로드
news_df = pd.read_csv(news_path, sep='\t', header=0)
news_df.columns = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Entity', 'Metadata']

# 카테고리와 서브카테고리를 결합하여 새로운 콘텐츠 생성
news_df['CategoryContent'] = news_df['Category'].fillna('') + " " + news_df['SubCategory'].fillna('')

# 결측값 제거
news_df = news_df.dropna(subset=['CategoryContent']).reset_index(drop=True)

# Count Vectorizer를 사용해 카테고리 기반 벡터화
vectorizer = CountVectorizer()
category_matrix = vectorizer.fit_transform(news_df['CategoryContent'])

# 코사인 유사도 계산
cosine_sim = cosine_similarity(category_matrix, category_matrix)

# 특정 기사를 기준으로 유사한 기사 추천
def recommend_by_category(target_index, t=5, threshold=0.1):
    similarity_scores = {idx: score for idx, score in enumerate(cosine_sim[target_index]) if idx != target_index and score > threshold}
    
    # 상위 t개의 추천 기사 선택
    sorted_recommendations = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = sorted_recommendations[:t]
    
    # 유사도 점수에 따라 상위 t개의 (뉴스 기사 ID, 유사도 점수)의 튜플을 리스트 형식으로 반환
    return [(news_df.iloc[news_id]['NewsID'], score) for news_id, score in top_recommendations]

'''
# 특정 기사를 기준으로 유사한 기사 추천
target_index = 0  # 추천 기준이 될 뉴스의 인덱스
similarity_scores = list(enumerate(cosine_sim[target_index]))

# 유사도 점수로 정렬
similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

# 동일한 카테고리에 속하는 상위 5개 추천
threshold = 0.3  # 임계값 설정 (0.3 이상인 추천만 포함)
top_recommendations = [score for score in similarity_scores if score[1] > threshold and score[0] != target_index][:5]

# 추천 결과 출력
recommendations = []
for idx, score in top_recommendations:
    recommendations.append({
        'NewsID': news_df.iloc[idx]['NewsID'],
        'Title': news_df.iloc[idx]['Title'],
        'Category': news_df.iloc[idx]['Category'],
        'SubCategory': news_df.iloc[idx]['SubCategory'],
        'Similarity': score
    })

# 결과를 데이터프레임으로 정리
recommendation_df = pd.DataFrame(recommendations)
print(recommendation_df)
'''
