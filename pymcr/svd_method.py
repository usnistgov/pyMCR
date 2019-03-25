import numpy as np
from numpy.linalg import svd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class svd_method:
	def __init__(self, D, nSVD):
		"""Perform SIMPLISMA
		
		D = data matrix
		
		nSVD = maximum number of SVD components
		
		"""
		print('Performing SVD')
		
	def svd(D, nSVD): 
		D_0mean = D - D.mean(0)
		
		U, s, Vh = np.linalg.svd(D_0mean, full_matrices=False)
		
		pca = PCA(n_components=nSVD)
		pca.fit(D)
		D_transformed = pca.transform(D)
		
		eigens = pca.singular_values_
		variance = pca.explained_variance_
		n_sample = D.shape[0]
		
		total_var = (D_0mean**2).sum()/(n_sample-1)
		vs = np.zeros(nSVD)
		
		for i in range(nSVD):
			Xi = U[:,i].reshape(-1, 1)*s[i]@Vh[i].reshape(1, -1)
			vs[i] = np.sum(Xi**2)/(n_sample-1)
		explained = vs
		
		total_var = (D_0mean**2).sum()/(n_sample-1)
		rs = np.zeros(nSVD)
		for i in range(nSVD):
			Xi = U[:,i].reshape(-1, 1)*s[i]@Vh[i].reshape(1, -1)
			rs[i] = np.sum(Xi**2)/((n_sample-1)*total_var)
		explained_variance_ratio = rs
		
		plt.subplot(1, 2, 1)
		plt.plot(np.asarray(range(nSVD))+1, eigens, 'o-')
		plt.ylabel('Eigenvalues')
		plt.xlabel('Principle Component')
		plt.xticks(np.arange(0, nSVD+1, 2))
		
		plt.subplot(1, 2, 2)
		plt.plot(np.asarray(range(nSVD))+1, np.cumsum(explained_variance_ratio), 'o-')
		plt.ylabel('Explained Variance')
		plt.xlabel('Principle Component')
		plt.xticks(np.arange(0, nSVD+1, 2))
		
		plt.show()
		
		return eigens, explained_variance_ratio
