import streamlit as st
import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import ssl

# ignoring ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode =  ssl.CERT_NONE

# url = input('Enter url - ')
url = 'https://www.ktpress.rw/2021/01/nepad-has-remained-true-to-its-vision-president-kagame-on-20th-anniversary/'

all_articles = {}
documents = []


class Article:
    def __init__(self, title, url):
        self.title = title
        self.url = url
    
    def __str__(self):
        return 'Title: {}. URL: {} '.format(self.title, self.url)


def scrap(url, count):
    if count < 1:
        return

    html = urllib.request.urlopen(url, context=ctx).read()
    soup = BeautifulSoup(html, 'html.parser')

    tags = soup('a')

    articles = []

    for tag in tags:
        # name = tag.get('title', None)
        name = tag.contents[0]
        url = tag.get('href', None)

        if url.startswith('https://www.ktpress.rw/2021') or url.startswith('https://www.ktpress.rw/2020'):
            articles.append(Article(name, url))

    for article in articles:
        if not article.url in all_articles:
            # print(article) 
            if not str.isspace(str(article.title)):
                documents.append(str(article.title))
            all_articles[article.url] = article
            scrap(article.url, count-1)


st.header('Scrapping news articles')
scrap(url, 1)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

st.write('Documents scrapped')
st.write(documents)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)


true_k = st.number_input('Number of clusters', 2, 10, 4)
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

st.header("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    st.header("Cluster {}:".format(i)),
    for ind in order_centroids[i, :10]:
        st.write(terms[ind]),
    # st.write('\n')
