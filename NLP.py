
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDFV, CountVectorizer as CV
from sklearn.linear_model import LogisticRegression as LR, SGDClassifier as SGD
from sklearn.metrics import classification_report as CR, confusion_matrix as CM, accuracy_score as AS
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.preprocessing import LabelBinarizer as LB
from bs4 import BeautifulSoup as BS
from nltk.corpus import stopwords as SW
from nltk.tokenize.toktok import ToktokTokenizer as TTK
from nltk.stem import PorterStemmer as PS
from wordcloud import WordCloud as WC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import re
import os

warnings.filterwarnings('ignore')

path = '../input/IMDB Dataset.csv'
data = pd.read_csv(path)
n_train, n_test = 40000, 10000

train_data, test_data = data[:n_train], data[n_train:]
train_reviews, train_labels = train_data['review'], train_data['sentiment']
test_reviews, test_labels = test_data['review'], test_data['sentiment']

def preprocess_text(text):
    text = BS(text, "html.parser").get_text()
    text = re.sub('\[[^]]*\]', '', text)
    text = re.sub(r'[^a-zA-z0-9\s]', '', text)
    stemmer = PS()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def remove_stopwords(text, tokenizer, stopwords_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords_list]
    return ' '.join(filtered_tokens)

tokenizer = TTK()
stopwords_list = SW.words('english')
data['review'] = data['review'].apply(preprocess_text)
data['review'] = data['review'].apply(lambda x: remove_stopwords(x, tokenizer, stopwords_list))

norm_train_reviews = data.review[:n_train]
norm_test_reviews = data.review[n_train:]

cv = CV(min_df=0, max_df=1, binary=False, ngram_range=(1,3))
cv_train_reviews = cv.fit_transform(norm_train_reviews)
cv_test_reviews = cv.transform(norm_test_reviews)

tv = TFIDFV(min_df=0, max_df=1, use_idf=True, ngram_range=(1,3))
tv_train_reviews = tv.fit_transform(norm_train_reviews)
tv_test_reviews = tv.transform(norm_test_reviews)

lb = LB()
sentiments = lb.fit_transform(data['sentiment'])
train_sentiments, test_sentiments = sentiments[:n_train], sentiments[n_train:]

lr_model = LR(penalty='l2', max_iter=500, C=1, random_state=42)
lr_bow = lr_model.fit(cv_train_reviews, train_sentiments)
lr_tfidf = lr_model.fit(tv_train_reviews, train_sentiments)

lr_bow_pred = lr_model.predict(cv_test_reviews)
lr_tfidf_pred = lr_model.predict(tv_test_reviews)

print("LR BOW Score:", AS(test_sentiments, lr_bow_pred))
print("LR TFIDF Score:", AS(test_sentiments, lr_tfidf_pred))

print(CR(test_sentiments, lr_bow_pred, target_names=['Positive','Negative']))
print(CR(test_sentiments, lr_tfidf_pred, target_names=['Positive','Negative']))

print(CM(test_sentiments, lr_bow_pred, labels=[1,0]))
print(CM(test_sentiments, lr_tfidf_pred, labels=[1,0]))
