import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import log10
from collections import Counter
import numpy as np
from scipy.spatial.distance import cosine

docs = [
    "Italian cuisine is characterized by its simple, fresh ingredients like tomatoes, olive oil, garlic and pasta.",
    "Japanese food often incorporates seafood, like sushi and sashimi made with fresh raw fish.",
    "Indian curries use a blend of spices including turmeric, cumin, and coriander along with meats or vegetables.",
    "Mediterranean diet is based on vegetables, grains, olive oil and moderate portions of meat and fish."
]

query = "indian spice meat"

dic_doc = {i: doc for i, doc in enumerate(docs)}

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def compute_tf(document):
    freq = {}
    for word in document:
        freq[word] = freq.get(word, 0) + 1
    tf_dict = {word: count / len(document) for word, count in freq.items()}
    return tf_dict

def compute_idf(documents):
    corpus = set()
    for document in documents:
        corpus.update(document)
    
    document_count = len(documents)
    word_counts = Counter()
    for document in documents:
        word_counts.update(set(document))
    
    idf_dict = {word: log10(float(document_count) / count) for word, count in word_counts.items()}
    return idf_dict

def compute_tfidf(document, documents, vocab):
    tf_dict = compute_tf(document)
    idf_dict = compute_idf(documents)
    tfidf_vector = [tf_dict.get(word, 0) * idf_dict.get(word, 0) for word in vocab]
    return tfidf_vector

vocab = set()
preprocessed_documents = [preprocess(doc) for doc in docs]
for document in preprocessed_documents:
    vocab.update(document)

print(vocab)
processed_query = preprocess(query)

tfidf_documents = [np.array(compute_tfidf(document, preprocessed_documents, vocab)) for document in preprocessed_documents]
tfidf_query = np.array(compute_tfidf(processed_query, preprocessed_documents, vocab))

print(tfidf_query)

max_sim = float('-inf')
results = {}

for i, doc_vec in enumerate(tfidf_documents):
    sim = 1 - cosine(tfidf_query, doc_vec)
    if sim > max_sim:
        results["output"] = i
        max_sim = sim

print(f'The most matching document is\n{dic_doc[results["output"]]}')

