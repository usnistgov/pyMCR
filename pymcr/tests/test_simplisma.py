import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy.optimize as optimize

#generate some sample data array
#create random normalised gaussian functions

#Number of Spectral Components
nPure = 5
#Allowed Noise Percentage
noise = 5	

x0 = np.zeros(nPure)
sigma = np.zeros(nPure)
for i in range(nPure):
	x0[i] = random.uniform(-100, 100)
	sigma[i] = random.uniform(3, 25)

x = np.linspace(start = -120, stop = 120, num = 2000)

gx = np.zeros((len(x),5))
plt.subplot(3, 1, 1)
plt.subplots_adjust(left=0.1, bottom=0.075, right=0.95, top=0.9, wspace=0.2, hspace=0.5)


for i in range(5):
	gx[:,i] = np.exp(-(x-x0[i])**2/(2*sigma[i]**2))/np.sqrt(2*np.pi*sigma[i]**2)
	plt.plot(gx[:,i])
	plt.title('Real Components')

#create array with random normalised linear combination of gaussian functions
nspec = 200
D = np.zeros((len(x), nspec))
idx = list(range(nPure))

for i in range(nspec):
	randj = np.zeros(nPure)
	random.shuffle(idx)
	for j in range(nPure):
		randj[j] = random.uniform(0, 1-np.sum(randj))
		D[:,i] = gx[:,idx[j]]*randj[j]

#Main Algorithm
def simplisma(D, nr, error):

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
		
	plt.subplot(3, 1, 2)
	plt.plot(S)
	plt.title('Estimate Components')
	
	C = np.dot(np.linalg.pinv(S), D)
	
	plt.subplot(3, 1, 3)
	for i in range(nr):
		plt.plot(C[i])
	plt.title('Concentrations')
	plt.show()
	
	return S, C

#Run Simplisma
S, C = simplisma(D, nPure, noise)

