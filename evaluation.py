'''
compare our result with ground truth
'''
import os
import pickle as pickle
import numpy as np
import pdb
import matplotlib.pyplot as plt
from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)
from Trj_class_and_func_definitions import *
from getGroundTruthObjDict import GTtrj
from getVehiclesPairs import get_Co_occur,get_Co_location



if __name__ == '__main__':

	"""load GT trj dictionary:"""
	if dataSource == 'NGSIM':
		filename = os.path.join(DataPathobj.pairpath,'GTtrjdictionary_'+dataSource+'_cam4.p')
	else:
		filename = os.path.join(DataPathobj.pairpath,'GTtrjdictionary_'+dataSource+'.p')

	GTtrjdic = pickle.load(open(filename,'rb'))

	"""load our system result trj dictionary:"""
	VehicleObjDic = pickle.load(open(os.path.join(DataPathobj.pairpath,'VehicleObjDic.p'),'rb'))



	"""find the same time, assign the nearest one to the GT"""



	"""nearest neighbour"""
	cap = cv2.VideoCapture(DataPathobj.video)
	subSampRate = int(np.round(cap.get(cv2.cv.CV_CAP_PROP_FPS)))/Parameterobj.targetFPS

	vehicleCandidates={}  ## from vehilce candidates choose the nearest one to assign to GT
	for ii in GTtrjdic.keys():
		vehicleCandidates[ii] = []
		# print min(GTtrjdic[ii].frame),  max(GTtrjdic[ii].frame)
		for jj in VehicleObjDic.keys():
			[coorccurStatus, cooccur_ran, cooccur_IDs] = get_Co_occur(GTtrjdic[ii], VehicleObjDic[jj],subSampRate)
			if coorccurStatus:
				print len(cooccur_ran)
				[co1X, co2X, co1Y, co2Y] = get_Co_location(cooccur_ran,GTtrjdic[ii], VehicleObjDic[jj],subSampRate, isWrite=False) #get xy and write to file
				distance = np.mean(np.sqrt((co1X-co2X)**2+(co1Y-co2Y)**2))
				vehicleCandidates[ii].append([jj,cooccur_ran,distance])













