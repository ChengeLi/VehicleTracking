
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
		# filename = os.path.join(DataPathobj.pairpath,'GTtrjdictionary_'+dataSource+'_cam4.p')
		filename = os.path.join(DataPathobj.pairpath,'GTtrjdictionary_'+dataSource+'.p')
	else:
		filename = os.path.join(DataPathobj.pairpath,'GTtrjdictionary_'+dataSource+'.p')
	GTtrjdic = pickle.load(open(filename,'rb'))


	"""load our system result trj dictionary:"""
	isClustered = True
	if isClustered:
		VehicleObjDic = pickle.load(open(os.path.join(DataPathobj.pairpath,'VehicleObjDic.p'),'rb'))
	else:
		VehicleObjDic = pickle.load(open(os.path.join(DataPathobj.pairpath,'notClustered_VehicleObjDic.p'),'rb'))



	# cap = cv2.VideoCapture(DataPathobj.video)
	# subSampRate = int(np.round(cap.get(cv2.cv.CV_CAP_PROP_FPS)))/Parameterobj.targetFPS
	# -- need to interpolate to full resolution, one location per frame
	subSampRate = 1

	vehicleCandidates={}  ## from vehilce candidates choose the nearest one to assign to GT
	for ii in GTtrjdic.keys():
		vehicleCandidates[ii] = []
		# print min(GTtrjdic[ii].frame),  max(GTtrjdic[ii].frame)
		for jj in VehicleObjDic.keys():
			[coorccurStatus, cooccur_ran, cooccur_IDs] = get_Co_occur(GTtrjdic[ii], VehicleObjDic[jj],subSampRate)
			if coorccurStatus:
				print len(cooccur_ran)
				[co1X, co2X, co1Y, co2Y] = get_Co_location(cooccur_ran,cooccur_IDs, GTtrjdic[ii],VehicleObjDic[jj],subSampRate, isWrite=False) 
				distance = np.mean(np.sqrt((co1X-co2X)**2+(co1Y-co2Y)**2))
				

				
				"""length within the bbox"""
				relativeID = cooccur_ran-GTtrjdic[ii].frame[0]
				overlap_upperLX = GTtrjdic[ii].fullGTupperL_list[relativeID,0]
				overlap_upperLY = GTtrjdic[ii].fullGTupperL_list[relativeID,1]

				overlap_lowerRX = GTtrjdic[ii].fullGTLowerR_list[relativeID,0]
				overlap_lowerRY = GTtrjdic[ii].fullGTLowerR_list[relativeID,1]



				plt.plot(GTtrjdic[ii].fullGTupperL_list[:,0],GTtrjdic[ii].fullGTupperL_list[:,1])
				plt.plot(GTtrjdic[ii].fullGTLowerR_list[:,0],GTtrjdic[ii].fullGTLowerR_list[:,1])

				plt.draw()
				# pdb.set_trace()
				plt.show()
				# plt.plot(co1X,co1Y,color = 'r')
				# plt.plot(co2X,co2Y,color = 'g')
				# plt.draw()
				# plt.show()
				# pdb.set_trace()

				# cnt = [np.int32([[np.minimum(overlap_upperLX[0],overlap_lowerRX[0]),np.minimum(overlap_upperLY[0],overlap_lowerRY[0])], \
				# 	[np.maximum(overlap_upperLX[0],overlap_lowerRX[0]),np.minimum(overlap_upperLY[0],overlap_lowerRY[0])],\
				# 	[np.maximum(overlap_upperLX[0],overlap_lowerRX[0]),np.maximum(overlap_upperLY[0],overlap_lowerRY[0])], \
				# 	[np.minimum(overlap_upperLX[0],overlap_lowerRX[0]),np.maximum(overlap_upperLY[0],overlap_lowerRY[0])]])]

				cnt = [np.int32([[overlap_upperLX[0],overlap_upperLY[0]], \
					[overlap_lowerRX[0],overlap_upperLY[0]],\
					[overlap_lowerRX[0],overlap_lowerRY[0]], \
					[overlap_upperLX[0],overlap_lowerRY[0]]])]


				if len(cnt)>0:
					cap = cv2.VideoCapture(DataPathobj.video)
					st,frame = cap.read()
					# cv2.drawContours( frame, np.int16(cnt), -1, (0, 255, 0), 3 )

					cv2.drawContours( frame, cnt, 0  , (0, 255, 0), 3 )

					cv2.imshow('cnt',frame)
					# cv2.waitKey(0)
				pdb.set_trace()



				# withinBBox = (co2X<np.maximum(overlap_upperLX,overlap_lowerRX))*\
				# (co2X>np.minimum(overlap_upperLX,overlap_lowerRX))* \
				# (co2Y<np.maximum(overlap_upperLY,overlap_lowerRY))* \
				# (co2Y>np.minimum(overlap_upperLY,overlap_lowerRY))
				# vehicleCandidates[ii].append([jj,len(cooccur_ran),distance,np.sum(withinBBox)])
				# pdb.set_trace()



	"""find the same time, assign the nearest one to the GT"""
	"""nearest neighbour"""
	vehicleCandidates_reordered ={}

	for ii in GTtrjdic.keys():
		vehicleCandidates_reordered[ii] = []
		overlapInd = np.argsort( np.array(vehicleCandidates[ii])[:,1])[::-1]
		distInd    = np.argsort( np.array(vehicleCandidates[ii])[:,2])
		withinInd  = np.argsort( np.array(vehicleCandidates[ii])[:,3])[::-1]
		# np.array(vehicleCandidates[ii])[distInd]
		# np.array(vehicleCandidates[ii])[overlapInd]

		nearestID = np.array(vehicleCandidates[ii])[distInd[:]][:,0] #5 nearest
		nearestID_inBB = np.array(vehicleCandidates[ii])[withinInd[:]][:,0] #5 nearest

		vehicleCandidates_reordered[ii] = nearestID

		# plt.figure()
		plt.plot(GTtrjdic[ii].fullGTupperL_list,GTtrjdic[ii].fullGTLowerR_list,color ='g')
		# plt.plot(GTtrjdic[ii].fullxTrj,GTtrjdic[ii].fullyTrj,color='r')
		# plt.plot(GTtrjdic[ii].GTupperL_list,GTtrjdic[ii].GTLowerR_list,color ='g')
		# plt.plot(GTtrjdic[ii].xTrj,GTtrjdic[ii].yTrj,color='r')
		# for ind in vehicleCandidates_reordered[ii] :
		# for ind in np.array(vehicleCandidates[ii])[distInd[:10]][:,0]:
		# for ind in np.array(vehicleCandidates[ii])[overlapInd[:3]][:,0]:
		for ind in nearestID_inBB[:2]:
			plt.plot(VehicleObjDic[ind].fullxTrj,VehicleObjDic[ind].fullyTrj,color='r')
			pdb.set_trace()
		plt.draw()
		plt.show()
		pdb.set_trace()
		plt.cla()
		plt.draw()

















