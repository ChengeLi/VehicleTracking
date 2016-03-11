import os
import platform
import glob as glob
import cv2
class parameter(object):

	def __init__(self,dataSource,VideoIndex):  # different VideoIndex for different videos

		self.trunclen  = 600
		self.targetFPS = 5 #subsampRate = FPS/targetFPS


		'for KLT tracker'
		self.klt_detect_interval = 5
		if dataSource == 'Johnson':
			self.useWarpped = False
			self.lk_params = dict(winSize=(5, 5), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),flags = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS) #maxLevel: level of pyramid
			self.feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=3, blockSize=3)  #qualityLevel, below which dots will be rejected

		""" canal st """
		if dataSource == 'DoT':
			self.useWarpped = True
			self.lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03)) 
			# self.feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=7,blockSize=7)  
			self.feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=3, blockSize=5)  # old jayst 



		'for fit_extrapolate filtering'
		self.minspdth = 5
		self.transth = 100*self.targetFPS  #100s
		self.livelong_thresh = 4   # chk if trj is long enough

		self.trjoverlap_len_thresh = 5
		if not self.useWarpped:
			self.dth    = 300 #??!!!!
			self.yspdth = 5 #y speed threshold
			self.xspdth = 5 #x speed threshold
		else:
			self.dth = 210 #mean+std
			self.yspdth = 40
			self.xspdth = 30


		'for PCA, DPGMM in subspace_clutering_merge.py'
        # project_dimension = int(np.floor(sub_index.size / 100) + 1)
        # sub_labels_DPGMM, model = ssc.clustering_DPGMM(n_components=int(np.floor(sub_index.size / 4) + 1), alpha=0.001)



















