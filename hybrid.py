import ContentBased
from collections import defaultdict

#사용할 땐 hybridRecommendation 함수만 사용하시면 돼용
class hybridRecommender:
  
  #hybridRecommendation에서 호출 
  def get_news_info(self, result):
    ids = [id_ for id_, _ in result]

    # id에 매칭되는 뉴스의 정보를 가져오기
    matched_news = ContentBased.news_df[ContentBased.news_df['NewsID'].isin(ids)]

    # NewsID를 기준으로 result의 순서에 맞게 정렬
    ordered_news = matched_news.set_index('NewsID').loc[ids].reset_index()

    # DataFrame을 딕셔너리 리스트로 변환
    news_info_list = ordered_news.to_dict(orient='records')

    return news_info_list
  
  #contents return 값, item-item return 값, metadata return 값 입력
  def hybridRecommendation(self, content, item, meta): 
    #기본 결과 형태 지정, {'id':[0, 0, 0]}
    dict_list = defaultdict(lambda: [0,0,0])

    #튜플 합쳐서 ('id', content_similarity, item_similarity, meta_similarity) 형태, 없는 정보는 0으로 
    for i, lst in enumerate([content, item, meta]):
      for id_, similarity in lst:
        dict_list[id_][i] = similarity 

    merged_list = [(id_, values[0], values[1], values[2]) for id_, values in dict_list.items()]

    #(id, sum(각 similarity * weight)) 형태로 변환
    weighted_list = [(id_, values[0] * 0.4 + values[1] * 0.3 + values[2] * 0.3) for id_, *values in merged_list]

    #sort
    sorted_weighted_list = sorted(weighted_list, key=lambda x: x[1], reverse=True)

    #상위 n개 추출 //지금 10
    result = sorted_weighted_list[:10]
    
    #id로 news info 가져오고 return
    result = self.get_news_info(result)
    
    return result
    