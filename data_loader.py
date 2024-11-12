# data_loader.py

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

class DataLoader:
    def __init__(self, news_path, behaviors_path):
        self.news_path = news_path
        self.behaviors_path = behaviors_path
        self.stop_words = set(stopwords.words('english'))
        self.news_df = None
        self.behaviors_df = None

    def download_nltk_resources(self):
        nltk.download('stopwords')
        nltk.download('punkt')

    def load_data(self):
        self.news_df = pd.read_csv(
            self.news_path, sep='\t', header=None,
            names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']
        )
        self.behaviors_df = pd.read_csv(
            self.behaviors_path, sep='\t', header=None,
            names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']
        )

    def preprocess_text(self, text):
        if pd.isnull(text):
            text = ''
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = word_tokenize(text, language='english')
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def preprocess_data(self):
        self.news_df['Title'] = self.news_df['Title'].fillna('')
        self.news_df['Abstract'] = self.news_df['Abstract'].fillna('')
        self.news_df['Content'] = self.news_df['Title'].astype(str) + ' ' + self.news_df['Abstract'].astype(str)
        self.news_df['Content'] = self.news_df['Content'].apply(self.preprocess_text)
        self.behaviors_df['History'] = self.behaviors_df['History'].fillna('')

    def get_data(self):
        self.download_nltk_resources()
        self.load_data()
        self.preprocess_data()
        return self.news_df, self.behaviors_df
