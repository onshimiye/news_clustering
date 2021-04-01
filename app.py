import streamlit as st

# Scrapping the web for news articles

# import urllib.request, urllib.parse, urllib.error
# from urllib.request import Request, urlopen
# from bs4 import BeautifulSoup
# import ssl

# # ignoring ssl
# ctx = ssl.create_default_context()
# ctx.check_hostname = False
# ctx.verify_mode =  ssl.CERT_NONE

# # url = 'https://www.ktpress.rw/2021/01/nepad-has-remained-true-to-its-vision-president-kagame-on-20th-anniversary/'
# url = 'http://en.igihe.com/'

# req = Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'})
# webpage = urlopen(req).read()

# all_articles = {}
# documents = []

# class Article:
#     def __init__(self, title, url):
#         self.title = title
#         self.url = url
    
#     def __str__(self):
#         return 'Title: {}. URL: {} '.format(self.title, self.url)

# headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}

# def scrap(url, count):
#     if count < 1:
#         return

#     # html = urllib.request.urlopen(url, context=ctx).read()
#     soup = BeautifulSoup(webpage, 'html.parser')
#     print(soup)
#     tags = soup('a')
#     print('tags')
#     print(tags)

#     print('tags contents')
#     # print(tags.contents)
#     business = []
#     arts_culture = []
#     sports = []
#     politics = []

#     for tag in tags:
#         # name = tag.get('title', None)
#         # name = tag.conte  nts[0]
#         url = tag.get('href', None)
#         print(tag)

#         if url.startswith('https://en.igihe.com/economy') or 'economy' in url:
#             business.append(url)
#         elif url.startswith('https://en.igihe.com/arts-culture') or 'arts-culture' in url:
#             arts_culture.append(url)
#         elif url.startswith('https://en.igihe.com/sports') or 'sports' in url:
#             sports.append(url)
#         elif url.startswith('https://en.igihe.com/politics') or 'politics' in url:
#             politics.append(url)


#     for article in business:
#         if not article.url in all_articles:
#             # print(article) 
#             if not str.isspace(str(article.title)):
#                 documents.append(str(article.title))
#             all_articles[article.url] = article
#             scrap(article.url, count-1)


# st.header('Scrapping news articles')
# depth = st.number_input('Recursion depth', 1, 20, 2)
# scrap(url, 1)


# st.write('Documents scrapped')
# st.write(documents)
# print(documents)

st.title('News Clustering System')

st.write('Here are our sources')

# Clustering

import pandas as pd

data = pd.read_csv('articles.csv')

st.write(data['Source'].unique())

st.write('You can get any news story/article clustered within its category by pasting its content below.')

story_input = st.text_input('Article content')
# Stemming the corpus

import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from collections import Counter
import nltk
import re
import sys

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stemmer = PorterStemmer()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
progress = 0 #for keeping track of where the function is


def stem(x):
  dirty = word_tokenize(x)
  tokens = []
  for word in dirty:
      if word.strip('.') == '': #this deals with the bug
        pass
      elif re.search(r'\d{1,}', word): #getting rid of digits
        pass
      else:
          tokens.append(word.strip('.'))
  
  tokens = pos_tag(tokens) #
  stems = ' '.join(stemmer.stem(key.lower()) for key, value in  tokens if value != 'NNP') #getting rid of proper nouns
  return stems


data['Stems'] = data.Content.apply(lambda x: stem(x))


# creating a vocabulary

from collections import Counter
all_words = Counter()
def count_everything(x, all_words):
  # global start
  # # global all_words
  # global progress
  x = x.split(' ')
  for word in x:
      all_words[word] += 1

for item in data.Stems:
  count_everything(item, all_words)


allwordsdf = pd.DataFrame(columns = ['words', 'count'])
allwordsdf['count'] = pd.Series(list(all_words.values()))
allwordsdf['words'] = pd.Series(list(all_words.keys()))
allwordsdf.index = allwordsdf['words']


from nltk.corpus import words
nltk.download('words')
nltkstems = [stemmer.stem(word) for word in words.words()] #stem the #words in the NLTK corpus so that they’re equivalent to the words in #the allwordsdf dataframe
nltkwords = pd.DataFrame() #make a new dataframe with the stemmed #NLTK words
nltkwords['words'] = nltkstems
allwordsdf = allwordsdf[allwordsdf['words'].isin(nltkwords['words'])] #keep only #those in the stemmed NLTK corpus

from sklearn.feature_extraction.text import TfidfVectorizer
stopwords = list(allwordsdf[(allwordsdf['count'] >= allwordsdf['count'].quantile(.995)) | (allwordsdf['count'] <= allwordsdf['count'].quantile(.4))]['words'])
vecvocab = list(allwordsdf[(allwordsdf['count'] < allwordsdf['count'].quantile(.995)) & (allwordsdf['count'] > allwordsdf['count'].quantile(.4))]['words'])
vec = TfidfVectorizer(stop_words = stopwords, vocabulary = vecvocab, tokenizer=None)


# transforming the dataframe
vec_matrix = vec.fit_transform(data.Stems)

from sklearn.cluster import KMeans
model = KMeans(n_clusters=4, verbose = 0)

model.fit(vec_matrix)

# adding prediction labels to the dataframe
data['Cluster'] = model.labels_


def pred(story):
    story = pd.DataFrame({'Index': [0],'Content':story})
    story['Stems'] = story.Content.apply(lambda x: stem(x))
    story_matrix = vec.fit_transform(story.Stems)

    pred = model.predict(story_matrix)
    pred = int(pred)

    similar = data.loc[(data['Cluster'] == pred)]
    similar = similar.head(10)

    print('Similar Articles')

    i = 0
    for index, row in similar.iterrows():
        i+=1
        st.write('{}.'.format(i), row['Title'], row['URL'])

    return similar[['Title', 'Content', 'URL' ,'Category', 'Source']]




pred(story_input)













# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.metrics import adjusted_rand_score

# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(documents)
# print(X)


# true_k = st.number_input('Number of clusters', 2, 10, 4)
# model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
# model.fit(X)

# st.header("Top terms per cluster:")
# order_centroids = model.cluster_centers_.argsort()[:, ::-1]
# terms = vectorizer.get_feature_names()
# for i in range(true_k):
#     st.header("Cluster {}:".format(i)),
#     for ind in order_centroids[i, :10]:
#         st.write(terms[ind]),
#     # st.write('\n')
