import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
news_path = 'news_select.tsv'
behaviors_path = 'aggregated_behaviors.tsv'

news_df = pd.read_csv(news_path, sep='\t', header=0)
behaviors_df = pd.read_csv(behaviors_path, sep='\t', header=None, names=["ImpressionID", "UserID", "Time", "History", "Impressions"])

# Rename columns for better understanding
news_df.columns = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Entity', 'Metadata']

# Drop rows where Title or Abstract are NaN
news_df = news_df.dropna(subset=['Title', 'Abstract'])

# Combine Title and Abstract for better content representation
news_df['Content'] = news_df['Title'].fillna('') + " " + news_df['Abstract'].fillna('')

# Handle missing data
news_df = news_df.dropna(subset=['Content'])  # Drop rows with missing content
news_df = news_df.reset_index(drop=True)

# Add category and subcategory for richer features
news_df['Content'] = news_df['Content'] + " " + news_df['Category'].fillna('') + " " + news_df['SubCategory'].fillna('')

# Reduce dataset size by sampling (if needed)
news_sampled_df = news_df.sample(n=5000, random_state=42).reset_index(drop=True)

# TF-IDF vectorization on the enhanced content
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(news_sampled_df['Content'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 특정 기사를 기준으로 유사한 기사 추천
def recommend_by_content(target_index, t=5, threshold=0.1):
    similarity_scores = {idx: score for idx, score in enumerate(cosine_sim[target_index]) if idx != target_index and score > threshold}
    
    # 상위 t개의 추천 기사 선택
    sorted_recommendations = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = sorted_recommendations[:t]
    
    # 유사도 점수에 따라 상위 t개의 (뉴스 기사 ID, 유사도 점수)의 튜플을 리스트 형식으로 반환
    return [(news_df.iloc[news_id]['NewsID'], score) for news_id, score in top_recommendations]


'''
# Example: Get top 5 similar articles for a specific article
target_index = 0  # Index of the target article
similarity_scores = list(enumerate(cosine_sim[target_index]))

# Sort by similarity score
similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

# Filter scores with a threshold
threshold = 0.2  # Only keep recommendations with similarity > 0.3
filtered_scores = [score for score in similarity_scores if score[1] > threshold and score[0] != target_index]

# Get top 5 recommendations
top_5_recommendations = filtered_scores[:5]


# Display recommendations
recommendations = []
for i, (idx, score) in enumerate(top_5_recommendations):
    recommendations.append({
        'NewsID': news_sampled_df.iloc[idx]['NewsID'],
        'Title': news_sampled_df.iloc[idx]['Title'],
        'Abstract': news_sampled_df.iloc[idx]['Abstract'],
        'Similarity': score
    })

# Convert to DataFrame for better visualization
recommendation_df = pd.DataFrame(recommendations)
print(recommendation_df)
'''