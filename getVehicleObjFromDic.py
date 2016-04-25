'''
get vehicle object from time, X, Y dictionaries (trj2dic.py)
for comparisons with the Ground truth vehicle obj
'''

import numpy as np
import glob as glob
import cPickle as pickle
import matplotlib.pyplot as plt
from Trj_class_and_func_definitions import *
from DataPathclass import *
DataPathobj = DataPath(dataSource, VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)
import pdb

if __name__ == '__main__':
	# useCC = True  #For johnson
	useCC = False
	isClustered = True
	if isClustered:
		if useCC:
			test_vctime = pickle.load( open(os.path.join(DataPathobj.dicpath,'CC_final_vctime_consecutive_frame.p'), "rb" ) )
			test_vcxtrj = pickle.load( open(os.path.join(DataPathobj.dicpath,'CC_final_vcxtrj.p'), "rb" ) )
			test_vcytrj = pickle.load( open(os.path.join(DataPathobj.dicpath,'CC_final_vcytrj.p'), "rb" ) )			
		
		else:
			# test_vctime = pickle.load( open(os.path.join(DataPathobj.dicpath,'final_vctime.p'), "rb" ) )
			test_vctime = pickle.load( open(os.path.join(DataPathobj.dicpath,'final_vctime_consecutive_frame.p'), "rb" ) )
			test_vcxtrj = pickle.load( open(os.path.join(DataPathobj.dicpath,'final_vcxtrj.p'), "rb" ) )
			test_vcytrj = pickle.load( open(os.path.join(DataPathobj.dicpath,'final_vcytrj.p'), "rb" ) )
	else: ## directly from smoothed trj
			test_vctime = pickle.load( open(os.path.join(DataPathobj.dicpath,'vctime_consecutive_frame000.p'), "rb" ) )
			test_vcxtrj = pickle.load( open(os.path.join(DataPathobj.dicpath,'vcxtrj_000.p'), "rb" ) )
			test_vcytrj = pickle.load( open(os.path.join(DataPathobj.dicpath,'vcytrj_000.p'), "rb" ) )




	savePath = DataPathobj.pairpath

	cap = cv2.VideoCapture(DataPathobj.video)
	fps = int(np.round(cap.get(cv2.cv.CV_CAP_PROP_FPS)))
	subSampRate = fps/Parameterobj.targetFPS
	obj_pair = TrjObj(test_vcxtrj,test_vcytrj,test_vctime,subSampRate = subSampRate)
	badkeys  = obj_pair.bad_IDs1+obj_pair.bad_IDs2
	pdb.set_trace()

	clean_vctime = {}
	clean_vcxtrj = {}
	clean_vcytrj = {}
	for key in test_vctime.keys():
		if key in badkeys:
			continue
		else: 
			clean_vctime[key] = test_vctime[key]
			clean_vcxtrj[key] = test_vcxtrj[key]
			clean_vcytrj[key] = test_vcytrj[key]

	print "trj remaining: ", str(len(clean_vctime))
	# rebuild this object using filtered data, should be no bad_IDs
	AllTrjObj = TrjObj(clean_vcxtrj,clean_vcytrj,clean_vctime,subSampRate = subSampRate)
	# pickle.dump(AllTrjObj,open(os.path.join(savePath,'AllTrjObj.p'),'wb'))
	
	"""construct vehicle obj dictionary from AllTrjObj"""
	VehicleObjDic = {}
	for loopVehicleID1 in AllTrjObj.globalID:
		VehicleObjDic[loopVehicleID1] = VehicleObj(AllTrjObj,loopVehicleID1)
	if useCC:
		pickle.dump(VehicleObjDic,open(os.path.join(savePath,'CCVehicleObjDic.p'),'wb'))
	else:
		pickle.dump(VehicleObjDic,open(os.path.join(savePath,'Clustered_VehicleObjDic.p'),'wb'))




















