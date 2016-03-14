# split each feature out and see its confusion matrix


## fit 2component GMM to the distance features
import numpy as np
from sklearn import mixture
import pickle as pickle
import pdb	
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


import matplotlib.mlab as mlab




np.random.seed(1)
# Generate random observations with two modes centered on 0
# and 10 to use for training.
# obs = np.concatenate((np.random.randn(100, 1),
# ...                       10 + np.random.randn(300, 1)))

def fitGMM(obs,bins):
	g = mixture.GMM(n_components=2)
	g.fit(obs) 
	print 'weights: ',g.weights_
	print 'mean:', g.means_
	print 'std: ', g.covars_ 
	# print g.predict([[0], [2], [9], [10]]) 
	# print np.round(g.score([[0], [2], [9], [10]]), 2)
	
	pdb.set_trace()
	Y = -g.score_samples(bins)[0]

	pdb.set_trace()

	y0 = mlab.normpdf(bins, g.means_[0], g.covars_[0])
	plt.plot(bins, g.weights_[0]*y0, 'g--')
	y1 = mlab.normpdf(bins, g.means_[1], g.covars_[1])
	plt.plot(bins, g.weights_[1]*y1, 'r--')
	plt.show()




sxdiffAll    = pickle.load(open('./sxdiffAll_johnson','rb'))
sydiffAll    = pickle.load(open('./sydiffAll_johnson','rb'))
mdisAll      = pickle.load(open('./mdisAll_johnson','rb'))
centerDisAll = pickle.load(open('./centerDisAll_johnson','rb'))
huedisAll    = pickle.load(open('./huedisAll_johnson','rb'))

nX, binsX, patches1 = plt.hist(sxdiffAll, 100, normed=1, facecolor='green', alpha=0.75)
nY, binsY, patches2 = plt.hist(sydiffAll, 100, normed=1, facecolor='green', alpha=0.75)
nD, binsD, patches3 = plt.hist(mdisAll, 100, normed=1, facecolor='green', alpha=0.75)
nCD, binsCD, patches4 = plt.hist(centerDisAll, 100, normed=1, facecolor='green', alpha=0.75)
nH, binsH, patches5 = plt.hist(huedisAll, 100, normed=1, facecolor='green', alpha=0.75)







fitGMM(sxdiffAll,binsX)








