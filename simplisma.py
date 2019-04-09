import numpy as np
import matplotlib.pyplot as plt
import random, sys
import scipy.optimize as optimize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Main Algorithm
class simplisma:
	def __init__(self,D, nPure, noise, lcf):
		"""Perform SIMPLISMA
		
		D = data matrix
		
		nPure = number of pure components

		noise = allowed noise percentage
		
		lcf = flag to perform constrained lcf 
		
		"""
		print('Performing Simplisma')
		
	def pure(D, nr, error, lcf):
	
		if lcf == True:
			xplt = 5
		else:
			xplt = 2
	
		def wmat(c,imp,irank,jvar):
			dm=np.zeros((irank+1, irank+1))
			dm[0,0]=c[jvar,jvar]
			
			for k in range(irank):
				kvar=np.int(imp[k])
				
				dm[0,k+1]=c[jvar,kvar]
				dm[k+1,0]=c[kvar,jvar]
				
				for kk in range(irank):
					kkvar=np.int(imp[kk])
					dm[k+1,kk+1]=c[kvar,kkvar]
			
			return dm
		
		nrow,ncol=D.shape
		
		dl = np.zeros((nrow, ncol))
		imp = np.zeros(nr)
		mp = np.zeros(nr)
		
		w = np.zeros((nr, ncol))
		p = np.zeros((nr, ncol))
		s = np.zeros((nr, ncol))
		
		error=error/100
		mean=np.mean(D, axis=0)
		error=np.max(mean)*error
		
		s[0,:]=np.std(D, axis=0)
		w[0,:]=(s[0,:]**2)+(mean**2)
		p[0,:]=s[0,:]/(mean+error)
	
		imp[0] = np.int(np.argmax(p[0,:]))
		mp[0] = p[0,:][np.int(imp[0])]
		
		l=np.sqrt((s[0,:]**2)+((mean+error)**2))
	
		for j in range(ncol):
			dl[:,j]=D[:,j]/l[j]
			
		c=np.dot(dl.T,dl)/nrow
		
		w[0,:]=w[0,:]/(l**2)
		p[0,:]=w[0,:]*p[0,:]
		s[0,:]=w[0,:]*s[0,:]
		
		print('purest variable 1: ', np.int(imp[0]+1), mp[0])
	
		for i in range(nr-1):
			for j in range(ncol):
				dm=wmat(c,imp,i+1,j)
				w[i+1,j]=np.linalg.det(dm)
				p[i+1,j]=w[i+1,j]*p[0,j]
				s[i+1,j]=w[i+1,j]*s[0,j]
				
			imp[i+1] = np.int(np.argmax(p[i+1,:]))
			mp[i+1] = p[i+1,np.int(imp[i+1])]
			
			print('purest variable '+str(i+2)+': ', np.int(imp[i+1]+1), mp[i+1])
			
		S=np.zeros((nrow, nr))
				
		for i in range(nr):
			S[0:nrow,i]=D[0:nrow,np.int(imp[i])]
		
		plt.subplot(xplt, 1, 1)
		plt.plot(S)
		
		C_u = np.dot(D.T, np.linalg.pinv(S.T))
		
		print(C_u.shape)
		
		plt.subplot(xplt, 1, 2)
		for i in range(nr):
			plt.plot(C_u[:,i])
		plt.ylabel('Unconstrained')
		
		if lcf == True:
			#Constrained linear combination
			def model(S, *wt0):
				s = np.zeros(S.shape[0])
				for i in range(len(wt0)):
					s = s + wt0[i]*S[:,i]
				return s/sum(wt0)
				
			CS = np.zeros(D.shape)
			wt0 = np.ones(S.shape[1])/(S.shape[1])
			
			lower = np.zeros(S.shape[1])
			upper = np.ones(S.shape[1])*1
			
			LOF = np.zeros(np.shape(D)[1])
			
			for i in range(np.shape(D)[1]):
				wt0 = np.abs(C_u[i,:])/sum(np.abs(C_u[i,:]))
				popt,pcov = optimize.curve_fit(model,S,D[:,i],p0=wt0, method='trf', bounds=(lower, upper))
				CS[:,i] = model(S, *popt)
				LOF[i] = 100*np.sqrt(np.sum(D[:,i]-CS[:,i])**2/np.sum(D[:,i])**2)
				sys.stdout.write("\r" + 'Percent complete (%): '+str(np.around(100*(i+1)/np.shape(D)[1], decimals=0))+' : '+str(np.around(LOF[i], decimals=2)))
				sys.stdout.flush()
				wt0 = popt
				
			C_c = np.dot(CS.T, np.linalg.pinv(S.T))
			
			plt.subplot(xplt, 1, 3)
			for i in range(nr):
				plt.plot(C_c[:,i])
			plt.ylabel('Constrained')
			
			xind = 6000
			
			plt.subplot(xplt, 1, 4)
			plt.plot(D[:,xind], '--k')
			for i in range(nr):
				plt.plot(C_c[:,i][xind]*S[:,i])
			plt.ylabel('Absorption')
				
			plt.subplot(xplt,1,5)
			plt.plot(LOF)
			plt.ylabel('LOF (%)')
			plt.show()

			return S, C_u, C_c
	
		else:
			plt.show()
			return S, C_u, 0
			
class svd:
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

