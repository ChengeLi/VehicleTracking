import os
import platform
import glob as glob
import pdb
import cv2

class DataPath(object):
	def __init__(self,dataSource,VideoIndex): 

		if platform.system()=='Darwin':   # on mac for test only, please ignore this....
			if dataSource == 'Johnson':
				self.sysPathHeader = '/Users/Chenge/Documents/github/AIG/Jay&Johnson/'
				self.videoPath = os.path.join(self.sysPathHeader,'./00115_ROI/')
				self.videoList = sorted(glob.glob(self.videoPath+'*.avi'))
				self.video = self.videoList[VideoIndex]
				self.videoTime = '00115_ROI'

			if dataSource == 'DoT':
				# self.sysPathHeader = '/Users/Chenge/Documents/github/AIG/DoT/'
				# self.videoPath = os.path.join(self.sysPathHeader,'./Convert3/')
				self.sysPathHeader = '/Users/Chenge/Desktop/stereo_vision/peopleCounter/'
				self.videoPath = os.path.join(self.sysPathHeader,'./data/')
				self.videoList = sorted(glob.glob(self.videoPath+'*.avi'))
				# self.video = '/Users/Chenge/Documents/github/AIG/DoT/Convert3/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms.avi'
				self.video = self.videoList[VideoIndex]
				# self.videoTime = self.video[-31:-4]
				self.videoTime = 'peopleCounter'

			if dataSource == 'NGSIM':
				self.sysPathHeader = '/Volumes/Transcend/US-101/US-101-RawVideo-0750am-0805am-Cam1234/'
				self.videoPath = os.path.join(self.sysPathHeader,'./00115_ROI/')
				self.videoList = sorted(glob.glob(self.videoPath+'*.avi'))
				self.video = '/Volumes/Transcend/US-101/US-101-RawVideo-0750am-0805am-Cam1234/sb-camera4-0750am-0805am.avi'
				self.videoTime = 'camera4'


		else:  # please start here and mofigy accordingly
			if dataSource == 'Johnson':
				self.sysPathHeader = '/media/My Book/CUSP/AIG/Jay&Johnson/'
				self.videoPath = os.path.join(self.sysPathHeader,'CUSPvideos/')
				self.videoList = sorted(glob.glob(self.videoPath+'*.avi'))
				self.video = self.videoList[VideoIndex]
				self.videoTime = self.video[47:-4]

			if dataSource == 'DoT':
				if os.getcwd()[:9] == '/scratch/':  # on HPC
					self.sysPathHeader = '/scratch/cl2840/CUSP/'
					# self.videoPath  = '/scratch/cl2840/CUSP/CanalBaxter/'
					# self.videoList = sorted(glob.glob(self.videoPath+'*.asf'))
					# self.video = self.videoList[VideoIndex]
					existingDirList = sorted(glob.glob('/scratch/cl2840/CUSP/2015-06*'))
					self.video = existingDirList[VideoIndex]
					self.videoTime = self.video[-14:]

				else:# on badminton linux
					## the 5th ave video
					# self.sysPathHeader = '/media/My Book/DOT Video/FifthAve/'
					# self.videoPath = '/home/chengeli/CUSP/AIG/DoT/ASF_files/'
					# self.videoList = sorted(glob.glob(self.videoPath+'*.asf'))  
					"""use .asf folder"""
					self.sysPathHeader = '/media/My Book/DOT Video/'
					self.videoPath = os.path.join(self.sysPathHeader,'Canal@Baxter/')
					self.videoList = sorted(glob.glob(self.videoPath+'*.asf'))
					"""use .avi folder"""
					# self.sysPathHeader = '/media/My Book/DOT Video/'
					# self.videoPath = os.path.join(self.sysPathHeader,'Canal@Baxter_avi/')
					# self.videoList = sorted(glob.glob(self.videoPath+'*.avi'))
					# self.video = self.videoList[VideoIndex]
					# self.videoTime = self.video[-31:-17]
			
			if dataSource == 'laurier':
				self.sysPathHeader = '/media/My Book/Saunier/'
				self.videoPath = '/home/chengeli/CUSP/AIG/Saunier/laurier/'
				self.videoList = sorted(glob.glob(self.videoPath+'*.avi'))
				self.video = self.videoList[VideoIndex]
				self.videoTime = 'ourAlgo_'+self.video[40:-4]


		self.cap = cv2.VideoCapture(self.video)
		self.DataPath = os.path.join(self.sysPathHeader,self.videoTime)
		if not os.path.exists(self.DataPath):
			os.mkdir(self.DataPath)
			
		
		self.imagePath = []
		self.kltpath = os.path.join(self.DataPath,"klt/")
		self.smoothpath = os.path.join(self.DataPath,"klt/smooth/")
		self.filteredKltPath = os.path.join(self.DataPath,"klt/filtered/")
		self.adjpath = os.path.join(self.DataPath,"adj/")
		self.sscpath = os.path.join(self.DataPath,"ssc/")
		self.dicpath = os.path.join(self.DataPath,"dic/")
		self.unifiedLabelpath = os.path.join(self.DataPath,"unifiedLabel/")
		self.pairpath = os.path.join(self.DataPath,"pair/")
		self.blobPath = os.path.join(self.DataPath,"incPCPmask/")
		self.visResultPath = os.path.join(self.DataPath,"visualization/")

		"""create folders"""
		pathList = [self.kltpath,self.smoothpath,self.filteredKltPath, self.adjpath,self.sscpath,\
		self.unifiedLabelpath,self.dicpath,self.pairpath,self.blobPath,self.visResultPath]
		for path in pathList:
			try:
				os.mkdir(path)
			except:
				print path,' exist, go on.'



"""Some Useful functions"""
from scipy.stats import norm
import numpy as np
# fitting Gaussian and get rid of the outlier(too large p3)
def fitGaussian(data):
	# Fit a normal distribution to the data:
	mu, std = norm.fit(np.array(data)[~np.isnan(data)])
	## Plot the histogram.
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



# def plotTrj(x,y,Trjchoice=[]):
# 	if Trjchoice==[]:
# 		Trjchoice=range(x.shape[0])

# 	plt.ion()
# 	plt.figure()
# 	im  = plt.imshow(np.zeros([508,710,3]))	
# 	for ii in range(0,len(Trjchoice),1):
# 		kk = Trjchoice[ii]
# 		xk = x[kk,:][x[kk,:]!=0]
# 		yk = y[kk,:][y[kk,:]!=0]
# 		if len(xk)>=5 and (min(xk.max()-xk.min(), yk.max()-yk.min())>2): # range span >=2 pixels
# 			# plt.plot(xk)
# 			# plt.plot(yk)
# 			plt.plot(xk, yk)
# 			# extraPolate(xk, yk)
# 			# x_fit = np.linspace(xk.min(), xk.max(), 200)
# 			# y_fit = pow(x_fit,3)*p3[ii,0] + pow(x_fit,2)*p3[ii,1] + pow(x_fit,1)*p3[ii,2]+ p3[ii,3]
# 			# plt.plot(x_fit, y_fit)
# 			plt.draw()
# 	plt.show()

