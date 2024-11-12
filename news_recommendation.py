import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from collections import Counter

# NLTK 데이터 다운로드
nltk.download('stopwords')
nltk.download('punkt')

# 불용어 설정
stop_words = set(stopwords.words('english'))

# 데이터셋 경로 설정
news_path = '/Users/yunseo/Desktop/ml/data/news.tsv'
behaviors_path = '/Users/yunseo/Desktop/ml/data/behaviors.tsv'

# 데이터 로드
news_df = pd.read_csv(news_path, sep='\t', header=None,
                      names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])
behaviors_df = pd.read_csv(behaviors_path, sep='\t', header=None,
                           names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])

# 데이터 확인
print("뉴스 데이터 샘플:")
print(news_df.head())
print("\n사용자 행동 데이터 샘플:")
print(behaviors_df.head())

# 데이터 전처리
news_df['Title'] = news_df['Title'].fillna('')
news_df['Abstract'] = news_df['Abstract'].fillna('')
news_df['Content'] = news_df['Title'].astype(str) + ' ' + news_df['Abstract'].astype(str)

def text_preprocessing(text):
    if pd.isnull(text):
        text = ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text, language='english')  # language 파라미터 추가
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

news_df['Content'] = news_df['Content'].apply(text_preprocessing)

# 전처리 결과 확인
print("\n전처리된 콘텐츠 샘플:")
print(news_df['Content'].head())

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, ngram_range=(1,2))
tfidf_matrix = tfidf_vectorizer.fit_transform(news_df['Content'])
print("\nTF-IDF 매트릭스의 크기:", tfidf_matrix.shape)

# 코사인 유사도 계산
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("코사인 유사도 매트릭스의 크기:", cosine_sim.shape)

# 기사 ID와 인덱스 매핑 생성
indices = pd.Series(news_df.index, index=news_df['NewsID']).drop_duplicates()
print("\n기사 ID와 인덱스 매핑 샘플:")
print(indices.head())

# 콘텐츠 기반 추천 시스템 함수 정의 (점수 포함)
def content_based_recommendations_with_scores(news_id, cosine_sim=cosine_sim):
    try:
        idx = indices[news_id]
    except KeyError:
        print(f"뉴스 ID {news_id}가 데이터셋에 존재하지 않습니다.")
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # 자기 자신 제외
    news_indices = [i[0] for i in sim_scores]
    scores = [score for (_, score) in sim_scores]
    return list(zip(news_df['NewsID'].iloc[news_indices], scores))

# 사용자 행동 데이터에서 읽은 기사 목록 추출
behaviors_df['History'] = behaviors_df['History'].fillna('')

# 각 사용자의 읽은 기사 리스트 생성
user_histories = behaviors_df.groupby('UserID')['History'].apply(lambda x: ' '.join(x)).reset_index()

# 사용자-아이템 매트릭스 생성
count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '))
user_item_matrix = count_vectorizer.fit_transform(user_histories['History'])
print("\n사용자-아이템 매트릭스의 크기:", user_item_matrix.shape)

# 사용자 간 코사인 유사도 계산
user_sim = cosine_similarity(user_item_matrix)
print("사용자 간 코사인 유사도 매트릭스의 크기:", user_sim.shape)

# 사용자 ID와 인덱스 매핑 생성
user_indices = pd.Series(user_histories.index, index=user_histories['UserID'])
print("\n사용자 ID와 인덱스 매핑 샘플:")
print(user_indices.head())

# 협업 필터링 추천 시스템 함수 정의 (user_sim을 매개변수로 사용)
def collaborative_recommendations_with_scores(user_id, user_sim, top_n=10):
    try:
        idx = user_indices[user_id]
    except KeyError:
        print(f"사용자 ID {user_id}가 데이터셋에 존재하지 않습니다.")
        return []
    sim_scores = list(enumerate(user_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]  # 자기 자신 제외
    similar_users = [i[0] for i in sim_scores]
    similar_users_histories = user_histories.iloc[similar_users]['History']
    all_articles = ' '.join(similar_users_histories).split(' ')
    article_counts = Counter(all_articles)
    user_read_articles = user_histories[user_histories['UserID'] == user_id]['History'].values[0].split(' ')
    recommended_articles = [article for article, count in article_counts.most_common() if article not in user_read_articles][:top_n]
    # 점수는 등장 빈도로 설정
    scores = [article_counts[article] for article in recommended_articles]
    return list(zip(recommended_articles, scores))

# 하이브리드 추천 시스템 함수 정의 (가중치 기반)
def hybrid_recommendations(user_id, news_id, user_sim, alpha=0.5, top_n=10):
    # 콘텐츠 기반 추천과 점수 가져오기
    content_recs = content_based_recommendations_with_scores(news_id)
    # 협업 필터링 추천과 점수 가져오기
    collaborative_recs = collaborative_recommendations_with_scores(user_id, user_sim, top_n=top_n)
    
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

# 예시: 특정 기사에 대한 콘텐츠 기반 추천
sample_news_id = news_df['NewsID'].iloc[0]
recommended_news = content_based_recommendations_with_scores(sample_news_id)
print("\n추천된 기사 ID (콘텐츠 기반):")
print(recommended_news)

# 예시: 특정 사용자에 대한 협업 필터링 추천
sample_user_id = behaviors_df['UserID'].iloc[0]
recommended_articles = collaborative_recommendations_with_scores(sample_user_id, user_sim=user_sim, top_n=10)
print("\n추천된 기사 ID (협업 필터링):")
print(recommended_articles)

# 예시: 특정 사용자와 기사에 대한 하이브리드 추천
recommended_articles = hybrid_recommendations(sample_user_id, sample_news_id, user_sim=user_sim, alpha=0.6, top_n=10)
print("\n추천된 기사 ID (하이브리드):")
print(recommended_articles)

# Word2Vec 기반 추천
# 기사 콘텐츠를 토큰화
news_df['Tokens'] = news_df['Content'].apply(word_tokenize)

# Word2Vec 모델 훈련
w2v_model = Word2Vec(sentences=news_df['Tokens'], vector_size=100, window=5, min_count=2, workers=4)
print("\nWord2Vec 단어 벡터 예시:")
try:
    print(w2v_model.wv['news'])  # 'news' 단어가 Word2Vec 모델에 있는지 확인
except KeyError:
    print("'news' 단어는 Word2Vec 모델에 없습니다.")

def get_article_vector(tokens):
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)

news_df['Vector'] = news_df['Tokens'].apply(get_article_vector)

# 벡터 매트릭스 생성
article_vectors = np.vstack(news_df['Vector'].values)
print("\nWord2Vec 벡터 매트릭스의 크기:", article_vectors.shape)

# 코사인 유사도 계산
cosine_sim_w2v = cosine_similarity(article_vectors, article_vectors)

def w2v_recommendations(news_id, cosine_sim=cosine_sim_w2v):
    try:
        idx = indices[news_id]
    except KeyError:
        print(f"뉴스 ID {news_id}가 데이터셋에 존재하지 않습니다.")
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    news_indices = [i[0] for i in sim_scores]
    return news_df['NewsID'].iloc[news_indices]

# 예시: Word2Vec 기반 추천
recommended_news = w2v_recommendations(sample_news_id)
print("\n추천된 기사 ID (Word2Vec 기반):")
print(recommended_news)

# SentenceTransformer 기반 추천
model = SentenceTransformer('all-MiniLM-L6-v2')

# 기사 콘텐츠 임베딩 (메모리 효율성을 위해 배치 처리 권장)
batch_size = 64
embeddings = []
contents = news_df['Content'].tolist()
for i in range(0, len(contents), batch_size):
    batch = contents[i:i+batch_size]
    emb = model.encode(batch, show_progress_bar=True)
    embeddings.extend(emb)
news_df['BERT_Vector'] = embeddings

# 벡터 매트릭스 생성
bert_vectors = np.vstack(news_df['BERT_Vector'].values)
print("\nBERT 벡터 매트릭스의 크기:", bert_vectors.shape)

# 코사인 유사도 계산
cosine_sim_bert = cosine_similarity(bert_vectors, bert_vectors)

def bert_recommendations(news_id, cosine_sim=cosine_sim_bert):
    try:
        idx = indices[news_id]
    except KeyError:
        print(f"뉴스 ID {news_id}가 데이터셋에 존재하지 않습니다.")
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    news_indices = [i[0] for i in sim_scores]
    return news_df['NewsID'].iloc[news_indices]

# 예시: BERT 기반 추천
recommended_news = bert_recommendations(sample_news_id)
print("\n추천된 기사 ID (BERT 기반):")
print(recommended_news)
