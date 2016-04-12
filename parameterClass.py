import os
import platform
import glob as glob
import cv2
class parameter(object):

	def __init__(self,dataSource,VideoIndex):  # different VideoIndex for different videos

		self.trunclen  = 600
		self.targetFPS = 5 #subsampRate = FPS/targetFPS
		
		"""for KLT tracker"""
		self.klt_detect_interval = 5
		if dataSource == 'Johnson':
			self.useSBS = False
			self.useWarpped = False
			self.lk_params = dict(winSize=(5, 5), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),flags = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS) #maxLevel: level of pyramid
			self.feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=3, blockSize=3)  #qualityLevel, below which dots will be rejected
			
			"""for fit_extrapolate filtering"""
			self.loc_change = 5
			self.minspdth = 5
			self.transth = 100*self.targetFPS  #100s
			self.livelong_thresh = 0.5*self.targetFPS   # chk if trj is long enough, 1s

			"""for adj SBS"""
			self.trjoverlap_len_thresh = 0.5*self.targetFPS  #0.5 s
			self.nullDist_for_adj = 50#if dis>= this value, adj[i,j] will be set to 0 
			#a car len: ~=100 to 200
			self.nullXspd_for_adj_norm = 0.1
			self.nullYspd_for_adj_norm = 0.1
			self.nullBlob_for_adj = 30


			"""for spectral embedding, DPGMM in subspace_clutering_merge.py"""
			self.embedding_projection_factor = 20
			self.DPGMM_num_component_shirink_factor = 1000
			self.DPGMM_alpha = 0.1


			self.useMask = False #already masked out!
			self.adj_weight = [2,2,2,0,1]


		""" canal st """
		if dataSource == 'DoT':
			self.useSBS = False
			self.useWarpped = True
			self.lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03)) 
			# self.feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=7,blockSize=7)  
			self.feature_params = dict(maxCorners=1000, qualityLevel=0.2, minDistance=2, blockSize=3)
			self.targetFPS = 30


			"""for fit_extrapolate filtering"""
			self.loc_change = 3
			self.minspdth = 5
			self.transth = 100*self.targetFPS  #100s
			self.livelong_thresh = 0.5*self.targetFPS   # chk if trj is long enough, 1s

			"""for adj SBS"""
			self.trjoverlap_len_thresh = 0.5*self.targetFPS  #0.5 s
			self.nullDist_for_adj = 300#if dis>= this value, adj[i,j] will be set to 0 

			"""for spectral embedding, DPGMM in subspace_clutering_merge.py"""
			self.embedding_projection_factor = 10
			self.DPGMM_num_component_shirink_factor = 4
			self.DPGMM_alpha = 10

			self.useMask = True
			self.adj_weight = [2,2,2,0,0.8]

		if dataSource == 'laurier':
			self.useSBS = True
			self.useWarpped = False
			self.lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03)) 
			self.feature_params = dict(maxCorners=500, qualityLevel=0.4, minDistance=3, blockSize=5)  # old jayst 
			self.targetFPS = 30

			"""for fit_extrapolate filtering"""
			self.loc_change = 5
			self.minspdth = 5
			self.transth = 100*self.targetFPS  #100s
			self.livelong_thresh = 0.5*self.targetFPS   # chk if trj is long enough, 1s


			"""for adj SBS"""
			self.trjoverlap_len_thresh = 0.5*self.targetFPS  #0.5 s
			self.nullDist_for_adj = 300#if dis>= this value, adj[i,j] will be set to 0 

			"""for spectral embedding, DPGMM in subspace_clutering_merge.py"""
			self.embedding_projection_factor = 10
			self.DPGMM_num_component_shirink_factor = 10
			self.DPGMM_alpha = 10

			self.useMask = True
			self.adj_weight = [2,2,2,0,0.8]



		if dataSource == 'NGSIM':
			# self.useSBS = False
			self.useSBS = True
			self.useWarpped = False
			self.lk_params = dict(winSize=(5, 5), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03)) 
			# self.feature_params = dict(maxCorners=2000, qualityLevel=0.01, minDistance=2, blockSize=5)  #more points
			self.feature_params = dict(maxCorners=2000, qualityLevel=0.1, minDistance=2, blockSize=5)  # old jayst 

			self.targetFPS = 10

			"""for fit_extrapolate filtering"""
			self.loc_change = 0.5
			self.minspdth = 3
			self.transth = 3*self.targetFPS #don't allow stopping
			self.livelong_thresh = 0  ##at least 2, for speed


			"""for adj SBS"""
			# self.trjoverlap_len_thresh = 0.5*self.targetFPS  #0.5 s
			self.trjoverlap_len_thresh = 2
			self.nullDist_for_adj = 50#if dis>= this value, adj[i,j] will be set to 0 
			self.nullXspd_for_adj_norm = 0.2
			self.nullYspd_for_adj_norm = 0.1
			self.nullBlob_for_adj = 15


			"""for spectral embedding, DPGMM in subspace_clutering_merge.py"""
			self.embedding_projection_factor = 15
			self.DPGMM_num_component_shirink_factor = 1.1
			self.DPGMM_alpha = 1000

			self.useMask = False

			# self.adj_weight = [2,1,1,0,1]
			# self.adj_weight = [4,1,1,0,2]
			self.adj_weight = [1,1,1,0,1]

			# self.adj_weight = [4,1,1,0,0]
			# self.adj_weight = [0,0,0,0,1]


		self.clustering_choice = 'labels_DPGMM_'
		# self.clustering_choice = 'labels_spectral_'












