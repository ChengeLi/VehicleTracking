import os
import platform
import glob as glob
import pdb

class DataPath(object):
    # def __init__(self):
    #     if platform.system()=='Darwin':   # on mac
    #         self.sysPathHeader = '/Volumes/TOSHIBA/'
    #     else:   # on linux
    #         self.sysPathHeader = '/media/TOSHIBA/'

	def __init__(self,VideoIndex):  # different VideoIndex for different videos
		if platform.system()=='Darwin':   # on mac
			self.sysPathHeader = '/Volumes/TOSHIBA/'
		else:   # on linux
			# self.sysPathHeader = '/media/TOSHIBA/'
			if os.getcwd()[-3:] == 'AIG':  # on CUSP compute #/home/cusp/cl2840/CanalVideos/Canal@Baxter/avi
				self.sysPathHeader = '../CanalVideos/Canal@Baxter/'
			else:
				self.sysPathHeader = '/media/My Book/DOT Video/'


		# self.videoPath = os.path.join(self.sysPathHeader,'Canal @ Baxter/Canal St @ Baxter St - 96.106_2015-06-20_08h00min00s000ms.asf')
		self.videoPath = os.path.join(self.sysPathHeader,'avi/')
		#pdb.set_trace()
		print self.videoPath
		self.videoList = sorted(glob.glob(self.videoPath+'*.avi'))
		#pdb.set_trace()
		self.video     = self.videoList[VideoIndex]
		self.videoTime = self.video[-31:-17]

		self.DataPath = os.path.join(self.sysPathHeader,self.videoTime)
		
		if not os.path.exists(self.DataPath):
			os.mkdir(self.DataPath)

		self.imagePath = []
		self.kltpath = os.path.join(self.DataPath,"klt/")
		self.filteredKltPath = os.path.join(self.DataPath,"klt/filtered/")
		self.adjpath = os.path.join(self.DataPath,"adj/")
		self.dicpath = os.path.join(self.DataPath,"dic/")
		self.sscpath = os.path.join(self.DataPath,"ssc/")
		self.pairpath = os.path.join(self.DataPath,"pair/")
		self.unifiedLabelpath = os.path.join(self.DataPath,"unifiedLabel/")
		"""create folders"""

		# if not os.path.exists(self.kltpath)
		try:
			os.mkdir(self.kltpath)
			os.mkdir(self.filteredKltPath)
			os.mkdir(self.adjpath)
			os.mkdir(self.unifiedLabelpath)
			os.mkdir(self.dicpath)
			os.mkdir(self.sscpath)
			os.mkdir(self.pairpath)
		except:
			print "folder exist, go on."














