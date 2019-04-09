import numpy as np
import random
import matplotlib.pyplot as plt
from pymcr.simplisma import svd, simplisma

#generate some sample data D
#create random normalised gaussian functions

#Number of Spectral Components
nPure = 5
#maximum number of components for SVD
nSVD = 15
#Allowed Noise Percentage
noise = 5	
#manual
manual = False

x0 = np.zeros(nPure)
sigma = np.zeros(nPure)
for i in range(nPure):
	x0[i] = random.uniform(-100, 100)
	sigma[i] = random.uniform(3, 25)

x = np.linspace(start = -120, stop = 120, num = 2000)

gx = np.zeros((len(x),nPure))
plt.subplot(5, 1, 1)
plt.subplots_adjust(left=0.1, bottom=0.075, right=0.95, top=0.9, wspace=0.2, hspace=0.5)

for i in range(nPure):
	gx[:,i] = np.exp(-(x-x0[i])**2/(2*sigma[i]**2))/np.sqrt(2*np.pi*sigma[i]**2)
	plt.plot(gx[:,i])
	plt.title('Real Components')

#create D with random normalised linear combination of gaussian functions
nspec = 200
D = np.zeros((len(x), nspec))
idx = list(range(nPure))

C_r = np.zeros((nspec, nPure))

for i in range(nspec):
	randj = np.zeros(nPure)
	random.shuffle(idx)
	for j in range(nPure):
		if j < nPure-1:
			randj[j] = random.uniform(0, 1-np.sum(randj))
			D[:,i] = D[:,i]+(gx[:,idx[j]]*randj[j])
			C_r[i,j] = randj[j]
		elif j == nPure-1:
			randj[j] = 1-np.sum(randj)
			D[:,i] = D[:,i]+(gx[:,idx[j]]*randj[j])
			C_r[i,j] = randj[j]
	
#run SVD
eigens, explained_variance_ratio = svd.svd(D, nSVD)
nPure = np.int(input('Number of Principle Components for SIMPLISMA :'))

#Run Simplisma
S, C_u, _, _ = simplisma.pure(D, nPure, noise, False)

#Run Simplisma with constrained LCF
S, C_u, C_c, LOF = simplisma.pure(D, nPure, noise, True)




