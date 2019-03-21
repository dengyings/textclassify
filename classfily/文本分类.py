import os
import jieba
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import json
import logging


def get_logger():
    """
    创建日志实例
    """
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    logger = logging.getLogger("monitor")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = get_logger()


def normalize(corpus):
    texts = []
    for text in corpus:
        text =" ".join(jieba.lcut(text))
        texts.append(text)
    return texts


def read_data(filename):
    book_data = pd.read_csv(filename)
    book_titles = book_data['title'].tolist()
    book_content = book_data['content'].tolist()
    norm_book_content = normalize(book_content)
    return book_data,book_titles,norm_book_content


def get_features(book_content):
    # 提取 tf-idf 特征
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(book_content).astype(float)
    feature_names = vectorizer.get_feature_names()
    return feature_matrix,feature_names


def k_means(feature_matrix, num_clusters=10):
    km = KMeans(n_clusters=num_clusters,
                max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters


def get_data(clustering_obj, book_data,
                     feature_names, num_clusters,
                     topn_features=10):
    deta = {}
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    for i in range(num_clusters):
        deta[i] = {}
        deta[i]['cluster_num'] = i
        key_features = [feature_names[index]for index
                        in ordered_centroids[i, :topn_features]]
        deta[i]['key_features'] = key_features
        books = book_data[book_data['Cluster'] == i]['title'].values.tolist()
        deta[i]['books'] = books
    return deta


def write(item):
    with open("recolist.json", "ab") as f:
        text = json.dumps(dict(item), ensure_ascii=False) + '\n'
        f.write(text.encode('utf-8'))
        logger.info("writeOK")


def main(argc, argv, envp):
    book_data,book_titles, book_content = read_data('data/data.csv')
    feature_matrix, feature_names = get_features(book_content)
    km_obj, clusters = k_means(feature_matrix=feature_matrix, num_clusters=10)
    book_data['Cluster'] = clusters
    c = Counter(clusters)
    logger.info('c.items', c.items())
    item = get_data(clustering_obj=km_obj,
                                    book_data=book_data,
                                    feature_names=feature_names,
                                    num_clusters=10,
                                    topn_features=5)
    write(item)
    return None


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))