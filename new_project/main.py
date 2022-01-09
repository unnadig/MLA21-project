import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.parsing.preprocessing as gs
# image data
w = 50
W_array = np.zeros((1300,w**2))
j = 0
# iterate through all files
for file in os.listdir():
    if file.endswith(".tiff"):
        X = plt.imread(file)
        # 100 windows from each image
        for i in range(100):
            W = X[i:i+w,i:i+w].flatten()
            W_array[j*100 + i] = W
        
        j += 1

W_array = np.transpose(W_array)
# text data
corpus = []
dirnames = ['sci.crypt/', 'sci.med/', 'sci.space/', 'soc.religion.christian/']
for dirname in dirnames:
    for file in os.listdir('20_newsgroups/'+dirname):
        with open('20_newsgroups/'+dirname+file, encoding="utf8", errors='ignore') as f:
            l = 0
            for line in f:
                # remove opening lines of document
                if l > 15:
                    contents = f.read()
                    # preprocessing for iftdf
                    filtered_text = gs.remove_stopwords(contents)
                    #filtered_text = gs.strip_non_alphanum(filtered_text)
                    corpus.append(filtered_text)
                l += 1

vectorizer = TfidfVectorizer()
response = vectorizer.fit_transform(corpus)

def RP(X,k):
    d,N = X.shape
    R = np.random.randn(k,d)
    Y = np.dot(R,X)
    return Y

#Y = RP(W_array, 500)

def SRP(X):
    Y = 0
    return Y

def PCA(X):
    Y = 0
    return Y

def DCT(X):
    Y = 0
    return Y

def recErr(X,Y):
    err = 0
    return err

def cosSim(X,Y):
    err = 0
    return err
