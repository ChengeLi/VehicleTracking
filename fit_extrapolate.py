# preprocessing padding to start and end regions
import os
import math
import pdb,glob
import numpy as np
from scipy.io import loadmat,savemat
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
# from DataPathclass import *
# DataPathobj = DataPath(VideoIndex)

# define start and end regions

#Canal video's dimensions:
"""(528, 704, 3)
start :<=110,
end: >=500,"""

start_Y = 110;
end_Y   = 500;

matfilepath    = '/Users/Chenge/Desktop/testklt/'
matfiles       = sorted(glob.glob(matfilepath + 'klt_*.mat'))
start_position = 0 
matfiles       = matfiles[start_position:]


# for matidx,matfile in enumerate(matfiles):
matidx = 0
matfile = matfiles[matidx]
# print "Processing truncation...", str(matidx+1)
ptstrj = loadmat(matfile)
mask = ptstrj['trjID'][0]
x    = csr_matrix(ptstrj['xtracks'], shape=ptstrj['xtracks'].shape).toarray()
y    = csr_matrix(ptstrj['ytracks'], shape=ptstrj['ytracks'].shape).toarray()
t    = csr_matrix(ptstrj['Ttracks'], shape=ptstrj['Ttracks'].shape).toarray()

# if consecutive points are similar to each other, merge them, using one to represent
Ntrj = len(x)
plt.figure()
# for ii in range(0,x.shape[0],100):
# 	# plt.plot(x[ii,:][x[ii,:]!=0])
# 	# plt.plot(y[ii,:][y[ii,:]!=0])
# 	plt.plot(x[ii,:][x[ii,:]!=0],y[ii,:][y[ii,:]!=0])

# xspd = np.zeros((x.shape[0],x.shape[1]-1))
badTrj = []
goodTrj = []
p3 = []
for kk in range(Ntrj):
	xk = x[kk,:][x[kk,:]!=0]
	yk = y[kk,:][y[kk,:]!=0]
	xspd = np.diff(xk[xk!=0])
	xaccelerate = np.diff(xspd)
	yspd = np.diff(yk[yk!=0])
	yaccelerate = np.diff(yspd)
	# print sum(xspd)
	# print sum(accelerate)
	# if (np.sum(xspd)<=1e-2 and np.sum(xaccelerate)<=1e-1) or (np.sum(yspd)<=1e-2 and np.sum(yaccelerate)<=1e-1):
	# if len(xspd)<=3 and (np.sum(xspd)<=1e-2) and (np.sum(xaccelerate)<=1e-1):
	if ((np.abs(np.sum(xspd))<=1e-2) and (np.abs(np.sum(yspd))<=1e-2)) or ((np.max(xk)-np.min(xk)<=5) and (np.max(yk)-np.min(yk)<=5)):
	# if len(xspd)<=5:	
		# print "xspd",xspd
		# print "yspd",yspd
		badTrj.append(kk)
		# plt.plot(x[kk,:][x[kk,:]!=0],y[kk,:][y[kk,:]!=0])
		# pdb.set_trace()
	else:
		goodTrj.append(kk)
		# pdb.set_trace()
		p3.append(np.polyfit(x[kk,:][x[kk,:]!=0], y[kk,:][y[kk,:]!=0], 3))  # fit a poly line to the last K points

p3 = np.array(p3)

plt.ion()
plt.figure()
for ii in range(0,len(goodTrj),1):
	kk = goodTrj[ii]
	xk = x[kk,:][x[kk,:]!=0]
	yk = y[kk,:][y[kk,:]!=0]
	if len(xk)>10:
		# plt.plot(xk)
		# plt.plot(yk)
		# plt.plot(xk, yk)
		# smooth(xk, yk)
		# extraPolate(xk, yk)
		x_fit = np.linspace(xk.min(), xk.max(), 200)
		y_fit = pow(x_fit,3)*p3[ii,0] + pow(x_fit,2)*p3[ii,1] + pow(x_fit,1)*p3[ii,2]+ p3[ii,3]
		plt.plot(x_fit, y_fit)
		plt.draw()
plt.show()

outlierID =[]
for ii in range(p3.shape[1]):
	data = p3[:,ii]
	mu,std = fitGaussian(data)
	outlierID = outlierID + list(np.where(data>=mu+2*std)[0])+list(np.where(data<=mu-2*std)[0])
	print p3[outlierID,:]
outlierID = np.unique(outlierID)




# fitting Gaussian and get rid of the outlier(too large p3)
from scipy.stats import norm
def fitGaussian(data):
	# Fit a normal distribution to the data:
	mu, std = norm.fit(data)
	# Plot the histogram.
	# plt.hist(data, bins=25, normed=True, alpha=0.6, color='g')
	# Plot the PDF.
	# xmin, xmax = plt.xlim()
	# x = np.linspace(xmin, xmax, 100)
	# p = norm.pdf(x, mu, std)
	# plt.plot(x, p, 'k', linewidth=2)
	# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
	# plt.title(title)
	# plt.show()
	return mu, std













# extrapolate original trj
from scipy.interpolate import InterpolatedUnivariateSpline
def extraPolate(xk, yk):
	# positions to inter/extrapolate
	# y_extraPosistion = np.linspace(start_Y, end_Y, 2)
	y_extraPosistion = range(start_Y, end_Y, 2)
	# spline order: 1 linear, 2 quadratic, 3 cubic ... 
	order = 1
	# do inter/extrapolation
	spline = InterpolatedUnivariateSpline(yk, xk, k=order)
	x_extraPosistion = spline(y_extraPosistion)

	# example showing the interpolation for linear, quadratic and cubic interpolation
	# plt.figure()
	plt.plot(x_extraPosistion, y_extraPosistion)
	plt.draw()
	# pdb.set_trace()
	# for order in range(1, 4):
	#     s = InterpolatedUnivariateSpline(xi, yi, k=order)
	#     y = s(x)
	#     plt.plot(x, y)
	plt.show()



from scipy.interpolate import spline
from scipy.interpolate import interp1d

def smooth(xk, yk):
	# # x_smooth = np.linspace(xk.min(), xk.max(), 200)
	# # y_smooth = spline(xk, yk, x_smooth,order = 2)
	# y_smooth = np.linspace(yk.min(), yk.max(), 200)
	# # x_smooth = spline(yk, xk, y_smooth, order = 1)
	# s = InterpolatedUnivariateSpline(yk, xk, k=2)
	# x_smooth = s(y_smooth)
	# plt.plot(x_smooth, y_smooth, linewidth=1)

	f = interp1d(xk, yk, kind='linear', axis=-1, copy=True, bounds_error=True, fill_value=np.nan, assume_sorted=False)
	xnew = np.arange(xk.min(), xk.max(),1)
	ynew = f(xnew)
	plt.plot(xnew, ynew, linewidth=1)
	plt.draw()
	# pdb.set_trace()
	# return x_smooth, y_smooth



# k-means on polynomial coefs
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(5)
estimators = {'k_means_20': KMeans(n_clusters=20),
              'k_means_8': KMeans(n_clusters=8),
              'k_means_iris_bad_init': KMeans(n_clusters=30, n_init=1,
                                              init='random')}

fignum = 1
for name, est in estimators.items():
    est.fit(p3)
    labels = est.labels_
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    ax.scatter(p3[:100, 0], p3[:100, 1], p3[:100, 2], c=labels[:100].astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    fignum = fignum + 1













