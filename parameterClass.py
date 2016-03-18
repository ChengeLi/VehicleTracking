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
			self.useSBS = True
			self.useWarpped = False
			self.lk_params = dict(winSize=(5, 5), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),flags = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS) #maxLevel: level of pyramid
			self.feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=3, blockSize=3)  #qualityLevel, below which dots will be rejected
			self.embedding_projection_factor = 20
			self.DPGMM_num_component_shirink_factor = 10
			
			'for adj SBS'
			self.trjoverlap_len_thresh = 0.5*self.targetFPS  #0.5 s
			self.nullDist_for_adj = 300#if dis>= this value, adj[i,j] will be set to 0 
			#a car len: ~=100 to 200

		""" canal st """
		if dataSource == 'DoT':
			self.useSBS = True
			self.useWarpped = True
			self.lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03)) 
			# self.feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=7,blockSize=7)  
			self.feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=3, blockSize=5)  # old jayst 
			self.embedding_projection_factor = 10
			self.DPGMM_num_component_shirink_factor = 4

			
			'for adj SBS'
			self.trjoverlap_len_thresh = 0.5*self.targetFPS  #0.5 s
			self.nullDist_for_adj = 300#if dis>= this value, adj[i,j] will be set to 0 
			#a car len: ~=100 to 200


		if dataSource == 'laurier':
			self.useSBS = True
			self.useWarpped = False
			self.lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03)) 
			self.feature_params = dict(maxCorners=500, qualityLevel=0.4, minDistance=3, blockSize=5)  # old jayst 
			self.targetFPS = 30
			self.embedding_projection_factor = 10
			self.DPGMM_num_component_shirink_factor = 10

			
			'for adj SBS'
			self.trjoverlap_len_thresh = 0.5*self.targetFPS  #0.5 s
			self.nullDist_for_adj = 300#if dis>= this value, adj[i,j] will be set to 0 
			#a car len: ~=100 to 200


		if dataSource == 'NGSIM':
			self.useSBS = False
			self.useWarpped = False
			self.lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03)) 
			self.feature_params = dict(maxCorners=500, qualityLevel=0.4, minDistance=3, blockSize=5)  # old jayst 
			self.targetFPS = 10
			self.embedding_projection_factor = 30
			self.DPGMM_num_component_shirink_factor = 2
			
			'for adj SBS'
			self.trjoverlap_len_thresh = 0.5*self.targetFPS  #0.5 s
			self.nullDist_for_adj = 40#if dis>= this value, adj[i,j] will be set to 0 
			#a car len: ~=100 to 200


		'for fit_extrapolate filtering'
		self.minspdth = 5
		self.transth = 100*self.targetFPS  #100s
		self.livelong_thresh = 1*self.targetFPS   # chk if trj is long enough, 1s


		'for PCA, DPGMM in subspace_clutering_merge.py'
		self.embedding_projection_factor = 10
		self.DPGMM_num_component_shirink_factor = 4


		self.clustering_choice = 'labels_DPGMM'
		# self.clustering_choice = 'labels_spectral'













