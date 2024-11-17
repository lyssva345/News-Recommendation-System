import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#news.tsv 데이터에서 랜덤하게 1만개 행만 선택한 뒤 해당 데이터를 사용
news_df = pd.read_csv('news.tsv', sep='\t')
news_select_df = news_df.sample(n=10000, random_state=42)
news_select_df.to_csv('news_select.tsv', sep='\t', index=False)

news_select_df = pd.read_csv('news_select.tsv', sep='\t')
# 샘플링 한 1만개의 뉴스 ID set
selected_news_ids = set(news_select_df.iloc[:, 0])

# behaviors.tsv 파일 로드 후 필요없는 첫 번째, 세 번째 열 삭제 (인덱스와 날짜)
behaviors_df = pd.read_csv('behaviors.tsv', sep='\t', header=None)
behaviors_df = behaviors_df.drop(columns=[0, 2])

# 유저 아이디, 클릭한 기사 ID, 노출 기사 중 클릭 하거나-1, 안한-0 기사 ID
behaviors_df.columns = ['user_id', 'news_ids_col', 'news_ids_col_suffix']
# 빈 문자열이 있는 행을 먼저 삭제
behaviors_df = behaviors_df[behaviors_df['news_ids_col'].notna() & behaviors_df['news_ids_col_suffix'].notna()]

# 관심 있는 뉴스 ID들을 새 열로 추가
def create_interested_news_id(row):
    
    # new_ids_col 열에서 1만 개의 랜덤 뉴스 ID 필터링
    news_ids_in_col = set(row['news_ids_col'].split())
    selected_ids_in_col = news_ids_in_col.intersection(selected_news_ids)

    # news_ids_col_suffix 열에서 뒤에 -1이 붙은 뉴스 ID들만 필터링하고, 그 중에서 1만 개 뉴스 ID와 겹치는 값을 선택
    news_ids_in_col_suffix = {id.split('-')[0] for id in row['news_ids_col_suffix'].split() if id.endswith('-1')}
    selected_ids_in_suffix = news_ids_in_col_suffix.intersection(selected_news_ids)
    
    # 위 두 과정에서 선택된 기사 ID 합침
    interested_ids = selected_ids_in_col.union(selected_ids_in_suffix)
    return ' '.join(interested_ids)

# 유저가 관심있어 하는 기사 ID 값을 새로운 열 'interested_news_id'로 생성
behaviors_df['interested_news_id'] = behaviors_df.apply(create_interested_news_id, axis=1)
# 이제는 필요없는 열 삭제
behaviors_df = behaviors_df.drop(columns=['news_ids_col', 'news_ids_col_suffix'])
# 공백이 있는 경우 삭제
behaviors_df = behaviors_df[behaviors_df['interested_news_id'] != '']

# 유저 ID별로 'interested_news_id' 값을 합치고 중복을 제거 (기존 behavior 데이터는 유저 ID 가 중복되는 행이 존재했다)
aggregated_df = behaviors_df.groupby('user_id')['interested_news_id'].agg(lambda x: ' '.join(sorted(set(' '.join(x).split())))).reset_index()
# 결과를 새로운 TSV 파일로 저장. 해당 결과는 user_id 열과, 해당 유저가 관심있어 하는 뉴스 기사를 공백을 두고 나열한 interested_news_id 열로 구성되어 있다.
aggregated_df.to_csv('aggregated_behaviors.tsv', sep='\t', index=False)

# 유저 ID 목록과 뉴스 ID 목록을 추출
user_ids = aggregated_df['user_id'].values
news_ids = list(selected_news_ids)
# user-item 매트릭스 초기화 (행: 유저, 열: 뉴스 기사)
user_item_matrix = pd.DataFrame(0, index=user_ids, columns=news_ids)

# 각 유저의 관심 있는 뉴스 ID를 기반으로 user-item 매트릭스 업데이트
for _, row in aggregated_df.iterrows():
    user_id = row['user_id']
    interested_news_ids = row['interested_news_id'].split()  # 관심있는 뉴스 기사 ID 리스트로 변환
    # 관심 있는 뉴스 ID에 대해 해당 유저의 셀 값은 1로 설정
    user_item_matrix.loc[user_id, interested_news_ids] = 1

# user-item 매트릭스 저장
user_item_matrix.to_csv('user_item_matrix.csv')

# 뉴스 아이템 간의 유사도 계산
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# 아이템(뉴스)간 유사도 행렬 저장
item_similarity_df.to_csv('item_similarity_matrix.csv')
