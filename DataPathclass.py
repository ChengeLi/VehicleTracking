import os
import platform
import glob as glob

class DataPath(object):
    # def __init__(self):
    #     if platform.system()=='Darwin':   # on mac
    #         self.sysPathHeader = '/Volumes/TOSHIBA/'
    #     else:   # on linux
    #         self.sysPathHeader = '/media/TOSHIBA/'

	def __init__(self,VideoIndex):  # different VideoIndex for different videos
		if platform.system()=='Darwin':   # on mac
			self.sysPathHeader = '/Volumes/TOSHIBA/'
			self.videoPath = os.path.join(self.sysPathHeader,'Canal@Baxter/')
			self.videoList = sorted(glob.glob(self.videoPath+'*.asf'))
		else:   # on linux
			if os.getcwd()[-3:] == 'AIG':  # on CUSP compute
				self.sysPathHeader = '../CanalVideos/Canal@Baxter/'
				self.videoPath = os.path.join(self.sysPathHeader,'avi/')
				self.videoList = sorted(glob.glob(self.videoPath+'*.avi'))
			else:
				self.sysPathHeader = '/media/My Book/DOT Video/'
				self.videoPath = os.path.join(self.sysPathHeader,'Canal@Baxter/')
				self.videoList = sorted(glob.glob(self.videoPath+'*.asf'))


		self.video     = self.videoList[VideoIndex]
		self.videoTime = self.video[-31:-17]

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
		"""create folders"""
		pathList = [self.kltpath,self.smoothpath,self.filteredKltPath, self.adjpath,self.sscpath,self.unifiedLabelpath,self.dicpath,self.pairpath]
		for path in pathList:
			try:
				os.mkdir(path)
			except:
				print path,' exist, go on.'



"""Some Useful functions"""
from scipy.stats import norm
# fitting Gaussian and get rid of the outlier(too large p3)
def fitGaussian(data):
	# Fit a normal distribution to the data:
	mu, std = norm.fit(data)
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



def plotTrj(x,y,Trjchoice=[]):
	if Trjchoice==[]:
		Trjchoice=range(x.shape[0])

	plt.ion()
	plt.figure()
	im  = plt.imshow(np.zeros([508,710,3]))	
	for ii in range(0,len(Trjchoice),1):
		kk = Trjchoice[ii]
		xk = x[kk,:][x[kk,:]!=0]
		yk = y[kk,:][y[kk,:]!=0]
		if len(xk)>=5 and (min(xk.max()-xk.min(), yk.max()-yk.min())>2): # range span >=2 pixels
			# plt.plot(xk)
			# plt.plot(yk)
			plt.plot(xk, yk)
			# extraPolate(xk, yk)
			# x_fit = np.linspace(xk.min(), xk.max(), 200)
			# y_fit = pow(x_fit,3)*p3[ii,0] + pow(x_fit,2)*p3[ii,1] + pow(x_fit,1)*p3[ii,2]+ p3[ii,3]
			# plt.plot(x_fit, y_fit)
			plt.draw()
	plt.show()





