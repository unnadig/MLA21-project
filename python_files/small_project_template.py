import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
# image data
X = plt.imread('1.tiff')
w = 50
W_array = np.zeros((1300,w**2))
for i in range(100):
    W = X[i:i+w,i:i+w].flatten()
    W_array[i] = W

W_array = np.transpose(W_array)
d = W.size
print(W_array.shape)
# text data
vectorizer = TfidfVectorizer()
response = vectorizer.fit_transform()

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