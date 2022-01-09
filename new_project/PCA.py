import numpy as np
from sklearn import random_projection
from sklearn import neighbors, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_random_state
from sklearn.datasets import fetch_20newsgroups_vectorized, fetch_20newsgroups
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
#Text Data
text_data = fetch_20newsgroups_vectorized().data[:3000]
n_samples, n_features = text_data.shape
n_components_range = np.array([200]) 
text_dists = euclidean_distances(text_data, squared=True).ravel()
nonzero = text_dists != 0
text_dists = text_dists[nonzero]
text_error_record=[]
for n_components in n_components_range:
    '''
    X_meaned = text_data - np.mean(text_data , axis = 0)
    cov_mat = np.cov(X_meaned , rowvar = False)
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    eigenvector_subset = sorted_eigenvectors[:,0:n_components]
    projected_dists = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
    '''
    rp = PCA(n_components=n_components)
    projected_data = rp.fit_transform(text_data.toarray())
    projected_dists = euclidean_distances(
        projected_data, squared=True).ravel()[nonzero]
    
    pca2_proj_back=rp.inverse_transform(projected_data)
    print(pca2_proj_back.shape)
    for j in range (n_components):
        #for i in range (data.shape[1]):
        #total_loss=LA.norm((data[j][i]-pca2_proj_back[j][i]),None)
        #print(text_data[j],pca2_proj_back[j])
        rmse = sqrt(mean_squared_error(text_data[j],pca2_proj_back[j]))
        text_error_record.append(rmse)
    #print(total_loss)
    
    plt.figure()
    plt.hexbin(text_dists, projected_dists, gridsize=100, cmap=plt.cm.PuBu)
    plt.xlabel("Pairwise squared distances in original space")
    plt.ylabel("Pairwise squared distances in projected space")
    plt.title("Pairwise distances distribution for reduced dimension=%d" %
              n_components)
    cb = plt.colorbar()
    cb.set_label('Sample pairs counts')

    plt.figure()
    plt.hist(rates, bins=100, density=True, stacked=True, range=(0., 2.), edgecolor='k')
    plt.xlabel("Squared distances rate: projected / original")
    plt.ylabel("Distribution of samples pairs")
    plt.title("Histogram of pairwise distance rates for reduced dimension=%d" %
              n_components)
    
    plt.figure()
    plt.plot(text_error_record,'r')
    plt.ylim([0,0.1])
    plt.xlabel("Reduced Dimension")
    plt.ylabel("Error Rate")
    plt.title("Error Plot PCA - Text Data")

plt.show()

# Image Data
img_data = fetch_olivetti_faces()
img_data = img_data.images.reshape((len(img_data.images), -1))[:]
n_samples, n_features = img_data.shape
n_components_range = np.array([300])
img_dists = euclidean_distances(img_data, squared=True).ravel()
nonzero = img_dists != 0
img_dists = img_dists[nonzero]
img_error_record=[]
for n_components in n_components_range:
    X_meaned = img_data - np.mean(img_data , axis = 0)
    cov_mat = np.cov(X_meaned , rowvar = False)
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    eigenvector_subset = sorted_eigenvectors[:,0:n_components]
    projected_dists = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
    rp = PCA(n_components=n_components)
    projected_data = rp.fit_transform(img_data)
    projected_dists = euclidean_distances(
        projected_data, squared=True).ravel()[nonzero]
    pca2_proj_back=rp.inverse_transform(projected_data)
    #print(pca2_proj_back)
    for j in range (n_components):
        #for i in range (data.shape[1]):
        #total_loss=LA.norm((data[j][i]-pca2_proj_back[j][i]),None)
        rmse = sqrt(mean_squared_error(img_data[j],pca2_proj_back[j]))
        img_error_record.append(rmse)
    plt.figure()
    plt.hexbin(img_dists, projected_dists, gridsize=100, cmap=plt.cm.PuBu)
    plt.xlabel("Pairwise squared distances in original space")
    plt.ylabel("Pairwise squared distances in projected space")
    plt.title("Pairwise distances distribution for reduced dimension=%d" %
              n_components)
    cb = plt.colorbar()
    cb.set_label('Sample pairs counts')

    plt.figure()
    plt.hist(rates, bins=100, density=True, stacked=True, range=(0., 2.), edgecolor='k')
    plt.xlabel("Squared distances rate: projected / original")
    plt.ylabel("Distribution of samples pairs")
    plt.title("Histogram of pairwise distance rates for reduced dimension=%d" %
              n_components)
    
    plt.figure()
    plt.plot(img_error_record,'r')
    plt.ylim([-0.1,0.1])
    plt.xlabel("Reduced Dimension")
    plt.ylabel("Error Rate")
    plt.title("Error Plot PCA - Image Data")

plt.show()