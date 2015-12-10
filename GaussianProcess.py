# This is the comparison for differenct covariance functions.
# Gained more understanding about GP from from http://www.cs.ubc.ca/~nando/540-2013/lectures.html 
# Used the plotting part from the demo code

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pdb
import csv
import glob as glob
import cv2
from scipy import interpolate




def Squared_exponential_kernel(a, b):
	""" squared exponential kernel """
	l      = 0.5
	tao    = 1
	sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
	kernel = tao**2*np.exp(-.5 * (1/(l**2)) * sqdist)
	parameters = [l, tao]
	return kernel, parameters


def Exponential_kernel(a, b):
	""" exponential kernel """
	l    = 4
	tao  = 1
	dist = np.zeros((len(a),len(b)))
	for ii in range(len(a)):
		dist[ii,:] = np.abs(b-a[ii]).reshape(len(b),)

	kernel     = tao**2*np.exp(-.5*(1/l) * dist)
	parameters = [l, tao]
	return kernel, parameters


def Spherical_kernel(a, b):
	""" Spherical kernel """
	tao   = 0.5
	theta = 1.1
	dist  = np.zeros((len(a),len(b)))
	for ii in range(len(a)):
		dist[ii,:] = np.abs(b-a[ii]).reshape(len(b),)

	cubdist = np.abs(dist**3)
	Kernel  = tao**2*(1-3*dist/2/theta + cubdist/2/(theta**2))
	Kernel_compact = Kernel*(np.array((dist<=theta))) # only have covariance values on the nearer locations
	parameters = [tao, theta]
	return Kernel_compact, parameters


def Linear_kernel(a, b):
	""" Linear kernel """
	sigma = 1
	tao   = 4
	c     = -4
	kernel = sigma**2+tao**2*np.dot(a-c,(b-c).T)
	parameters = [sigma, tao, c]
	return kernel, parameters


def draw_kernel(kernelNo):
	
	dist = np.linspace(-20,20,num = 200)

	if kernelNo == 1:
		l      = [0.5,1,4]
		tao    = [1,2,3]
		sqdist = dist**2
		kernel = lambda l, tao: tao**2*np.exp(-.5 * (1/(l**2)) * sqdist)
		fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
		ax1.plot(dist,kernel(l[0],tao[0]))
		ax2.plot(dist,kernel(l[1],tao[1]))
		ax3.plot(dist,kernel(l[2],tao[2]))

		caption = 'squared exponential kernel'+' l='+str(l)+' tao='+str(tao)
		plt.xlabel('distance',fontsize=14)
		ax1.set_title(caption)
		plt.savefig('./kernel1.jpg')

	if kernelNo == 2:
		l      = [0.5,1,4]
		tao    = [0.5,1,1.2]
		kernel = lambda l, tao: tao**2*np.exp(-.5*(1/l) * dist)
		fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
		ax1.plot(dist,kernel(l[0],tao[0]))
		ax2.plot(dist,kernel(l[1],tao[1]))
		ax3.plot(dist,kernel(l[2],tao[2]))

		caption = 'exponential kernel'+' l='+str(l)+' tao='+str(tao)
		plt.xlabel('distance',fontsize=14)
		ax1.set_title(caption)
		plt.savefig('./kernel2.jpg')

	if kernelNo == 3:
		tao   = [0.5,1,2]
		theta = [0.1,1.5,6]
		cubdist = np.abs(dist**3)
		Kernel_compact  = lambda tao, theta: tao**2*(1-3*dist/2/theta + cubdist/2/(theta**2))*(np.array((dist<=theta)))
		fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
		ax1.plot(dist,Kernel_compact(tao[0],theta[0]))
		ax2.plot(dist,Kernel_compact(tao[1],theta[1]))
		ax3.plot(dist,Kernel_compact(tao[2],theta[2]))
		caption = 'Spherical kernel'+' tao='+str(tao)+' theta='+str(theta)
		plt.xlabel('distance',fontsize=14)
		ax1.set_title(caption)
		plt.savefig('./kernel3.jpg')

	if kernelNo == 4:
		a = np.zeros((1,200))
		# b = np.random.uniform(-10, 10, size=(200,1))
		b = np.linspace(0,20,num = 200)

		dist = np.abs(a-b)

		sigma = [0.1,1,10]
		tao   = [1,2,4]
		c     = [-4,0,8]
		kernel = lambda sigma, tao,c: sigma**2+tao**2*(a-c)*(b-c)

		fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
		ax1.scatter(dist,kernel(sigma[0],tao[0],c[0]))
		ax2.scatter(dist,kernel(sigma[1],tao[1],c[1]))
		ax3.scatter(dist,kernel(sigma[2],tao[2],c[2]))
		caption = 'Linear kernel'+' sigma='+str(sigma)+' tao='+str(tao)+' c='+str(c)
		plt.xlabel('distance',fontsize=14)
		ax1.set_title(caption)
		plt.savefig('./kernel4.jpg')




def main(kernelNo,showBig = False):
	K,parameters = KernelOptions[kernelNo](X, X)
	L = np.linalg.cholesky(K + noise_var*np.eye(NumTraining))

	# compute mean for test points.
	K_star,parameters = KernelOptions[kernelNo](X, Xtest)
	Lk_star = np.linalg.solve(L, K_star)
	mu      = np.dot(Lk_star.T, np.linalg.solve(L, y))

	# compute variance at test points.
	K_star_star,parameters = KernelOptions[kernelNo](Xtest, Xtest)

	test_var = np.diag(K_star_star) - np.sum(Lk_star**2, axis=0)
	test_sigma = np.sqrt(test_var)


	# # draw samples from the prior at test points.
	L = np.linalg.cholesky(K_star_star + 1e-6*np.eye(NumTesting))
	f_prior = np.dot(L, np.random.normal(size=(NumTesting,10)))


	# # draw samples from the posterior at test points.
	L = np.linalg.cholesky(K_star_star + 1e-6*np.eye(NumTesting) - np.dot(Lk_star.T, Lk_star))
	f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(NumTesting,10)))
	

	if showBig:
		# plt.figure(1)
		# plt.clf()
		# plt.plot(X, y, 'r+', ms=20)
		# plt.plot(Xtest, f(Xtest), 'b-')
		# plt.gca().fill_between(Xtest.flat, mu-3*test_sigma, mu+3*test_sigma, color="#dddddd")
		# plt.plot(Xtest, mu, 'r--', lw=2)
		# figName = 'predictive'+str(kernelNo)+'.png'
		# plt.savefig(figName, bbox_inches='tight')
		# plt.title('Mean predictions plus 3 standard deviations'+str(parameters))
		# plt.axis([-10, 10, -3, 3])

		plt.figure(2)
		plt.clf()
		plt.plot(Xtest, f_prior)
		plt.title(str(NumTraining)+' samples from the GP prior')
		plt.axis([-10, 10, -3, 3])
		figName = 'prior'+str(kernelNo)+'.png'
		plt.savefig(figName, bbox_inches='tight')

		plt.figure(3)
		plt.clf()
		plt.plot(Xtest, f_post)
		plt.title(str(NumTraining)+' samples from the GP posterior '+str(parameters))
		plt.axis([-10, 10, -3, 3])
		figName = 'post'+str(kernelNo)+'.png'
		plt.savefig(figName, bbox_inches='tight')

		plt.show()




	# Three subplots sharing both x/y axes
	fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
	ax1.plot(X, y, 'r+', ms=20)
	ax1.plot(Xtest, f(Xtest), 'b-')
	ax1.fill_between(Xtest.flat, mu-3*test_sigma, mu+3*test_sigma, color="#dddddd")
	ax1.plot(Xtest, mu, 'r--', lw=2)
	ax1.set_title('Mean predictions plus 3 standard deviations'+str(parameters))
	ax1.axis([-10, 10, -3, 3])

	ax2.plot(Xtest, f_prior)
	ax2.set_title(str(NumTraining)+' samples from the GP prior')
	ax2.axis([-10, 10, -3, 3])

	ax3.plot(Xtest, f_post)
	ax3.set_title(str(NumTraining)+' samples from the GP posterior '+str(parameters))
	ax3.axis([-10, 10, -3, 3])

	figName = str(kernelNo)+'_'+str(parameters)+'.png'
	plt.savefig(figName, bbox_inches='tight')
	# fig.subplots_adjust(hspace=0)
	# plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)


def visGT():
	## visualize the Ground Truth
	# video_name = '../DoT/Convert3/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms.avi'
	# cap   = cv2.VideoCapture(video_name)
	# cap.set ( cv2.cv.CV_CAP_PROP_POS_FRAMES , 0)
	image_list = sorted(glob.glob('/Users/Chenge/Documents/github/AIG/DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/*.jpg'))

	color = np.array([np.random.randint(0,255) for _ in range(3*int(1000))]).reshape(int(1000),3)
	f      =  open('../rejayjohnsonintersectionpairrelationships/Canal_1.csv', 'rb')
	reader = csv.reader(f)

	# st,firstfrm = cap.read()
	firstfrm = cv2.imread(image_list[0])
	nrows    = int(np.size(firstfrm,0))
	ncols    = int(np.size(firstfrm,1))

	fig = plt.figure('vis')
	axL = plt.subplot(1,1,1)
	im  = plt.imshow(np.zeros([nrows,ncols,3]))
	plt.axis('off')


	dots           = []
	GTcenterXYList = []
	rectPtList     = []	
	NewCarAt = []
	frame_idx      = 0
	for kk in range(1000):
		temp = np.array(reader.next())
		if np.double(temp[0])<frame_idx: # new car
			color = np.array([np.random.randint(0,255)for _ in range(3*int(1000))]).reshape(int(1000),3)
			plt.cla()
			im  = plt.imshow(np.zeros([nrows,ncols,3]))
			plt.axis('off')
			NewCarAt.append(kk)

		frame_idx = np.int32(temp[0])
		GTcenterXY = np.double(temp[-2:])
		GTcenterXYList.append(GTcenterXY)
		pt1 = np.double(temp[1:3])
		pt2 = np.double(temp[3:5])
		rectPtList.append([pt1, pt2])
		# cap.set ( cv2.cv.CV_CAP_PROP_POS_FRAMES , frame_idx)
		# st,frame = cap.read()
		# frame = cv2.imread(image_list[frame_idx])
		# cv2.rectangle(frame, tuple(np.int16(pt1)), tuple(np.int16(pt2)),[0,255,0]) 
		# dots.append(axL.scatter(GTcenterXY[0], GTcenterXY[1], s=10, color=(color[100].T)/255.,edgecolor='none')) 
		# im.set_data(frame[:,:,::-1])
		# plt.draw()
		# plt.show()

		# plt.pause(0.00001)
		dots = []
		# name = '../GTfigure/'+str(np.int16(kk)).zfill(6)+'.jpg'
		# plt.savefig(name) ##save figure

	return np.array(GTcenterXYList), np.array(rectPtList),NewCarAt


if __name__ == '__main__':

	GTcenterXYList, rectPtList,NewCarAt = visGT()
	plt.figure('Ground Truth')
	for iii in range(len(NewCarAt)-1):
		plt.plot(GTcenterXYList[NewCarAt[iii]:NewCarAt[iii+1],0],GTcenterXYList[NewCarAt[iii]:NewCarAt[iii+1],1])


	pdb.set_trace()
	KernelOptions = {
		1 : Squared_exponential_kernel,
		2 : Exponential_kernel,
		3 : Spherical_kernel,
		4 : Linear_kernel,
	}

		
	NumTraining = 30         # number of training points.
	NumTesting  = 50         # number of test points.
	noise_var   = 0.00005    # noise variance.

	
	#the true unknown function
	f = lambda x: np.sin(0.9*x).flatten()+0.01*(x**2).flatten()
	# use manual GT as true function

	trueGTfunc = interpolate.interp1d(GTcenterXYList[:NewCarAt[0],0],GTcenterXYList[:NewCarAt[0],1]) # get the inverse transform
	

	f = trueGTfunc

	# get training samples
	# X     = np.random.uniform(-10, 10, size=(NumTraining,1))
	X     = np.random.uniform(-10, 10, size=(NumTraining,1))


	y     = f(X) + noise_var*np.random.randn(NumTraining) #noisy observations
	Xtest = np.linspace(-10, 10, NumTesting).reshape(-1,1)
	
	for kernelNo in range(1,5,1):
		main(kernelNo)














