# News-Recommendation-System


## Project Overview
This project aims to develop an article recommendation system that suggests related articles based on the topic of the article a user is currently reading. The goal is to encourage further content consumption and enhance user engagement, making it particularly useful for news websites and blogs to keep users on the platform longer through relevant recommendations.

## Datasets
The following datasets are used to train and test the recommendation system:

1. **[MIND Dataset (Microsoft News Dataset)](https://www.kaggle.com/datasets/arashnic/mind-news-dataset)**: Large-scale dataset from Microsoft News, containing user interactions, article topics, and descriptions, which are used for training the recommendation model.
2. **[Reddit News Dataset](https://www.kaggle.com/datasets/rootuser/worldnews-on-reddit)**: A collection of news articles and comments from Reddit, categorized by topics and interactions, suitable for testing the system’s performance in identifying relevant articles.


## Filtering Methods
The recommendation system employs several filtering techniques to personalize recommendations effectively:

1. **Content-Based Filtering**: Recommends articles by analyzing content similarity. Techniques like Term Frequency-Inverse Document Frequency (TF-IDF) and cosine similarity are used to identify articles with similar topics to the one the user is currently reading.
2. **Collaborative Filtering (optional)**: Analyzes user reading behavior to find similar users, recommending articles popular among users with similar interests. This includes both memory-based and model-based approaches for improved accuracy.
3. **Hybrid Filtering**: Combines content-based and collaborative filtering to provide personalized recommendations. By integrating user behavior data and article content similarity, the hybrid model can deliver balanced and accurate recommendations catering to both user preferences and article relevance.

## Machine Learning Models (Optional)
To enhance the recommendation system, several machine learning and NLP techniques are considered:

1. **Natural Language Processing (NLP)**: Methods like TF-IDF, Word2Vec, and BERT embeddings are used to represent article content and measure similarity between articles.
2. **Clustering**: Groups articles by topic to improve relevance in recommendations.
3. **Neural Networks**: Models like Recurrent Neural Networks (RNN) or Transformers are considered for capturing deeper semantic relationships between articles.

## Getting Started
### Prerequisites
- Python 3.14
- Libraries: pandas, numpy, nltk, sklearn, gensim, sentence-transformers

### Installation
Clone the repository and install the required libraries.

```bash
git clone https://github.com/lyssva345/News-Recommendation-System.git
cd News-Recommendation-System
pip install -r requirements.txt
