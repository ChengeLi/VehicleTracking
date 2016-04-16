
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



"""evaluation metrics"""
def metrics(vehicleCandidates_reorderedInd):
	detect       = 0
	overSegement = 0
	dist_list    = []
	for key in vehicleCandidates_reorderedInd.keys():
		if len(vehicleCandidates_reorderedInd[key][2])==0: ##not detected
			continue

		detect += (len(vehicleCandidates_reorderedInd[key][2])>0)
		overSegement += (len(vehicleCandidates_reorderedInd[key][2])>1)

		dist_ind = np.int16(vehicleCandidates_reorderedInd[key][5])
		dist_list.append( min(np.array(vehicleCandidates[key])[dist_ind,2]))

	Ntarget = (1.0*(len(vehicleCandidates_reorderedInd.keys())))
	missRate         = 1- detect/Ntarget
	overSegementRate = overSegement/Ntarget
	dist             = np.sum(dist_list)/Ntarget

	return missRate, overSegementRate, dist, dist_list



def plotGTonVideo(GTtrjdic, vehicleCandidates_reorderedInd=None):
	if dataSource == 'DoT':
		cap = cv2.VideoCapture(DataPathobj.video)
	elif dataSource == 'Johnson':
		cap = cv2.VideoCapture(DataPathobj.video)
	elif dataSource == 'NGSIM':
		# cap = cv2.VideoCapture('/Volumes/Transcend/US-101/US-101-ProcessedVideo-0750am-0805am-Cam1234/sb-camera4-0750am-0805am-processed.avi')
		cap = cv2.VideoCapture('/Volumes/Transcend/US-101/US-101-RawVideo-0750am-0805am-Cam1234/sb-camera4-0750am-0805am.avi')

	status, frame = cap.read()
	
	fig = plt.figure('vis GT on video')
	axL = plt.subplot(1,1,1)
	color = np.array([np.random.randint(0,255) for _ in range(3*len(GTtrjdic))]).reshape(len(GTtrjdic),3)
	plt.ion()
	dots = []
	for keyind in range(len(GTtrjdic.keys())):
	# for keyind in range(40,70,1):
		key = GTtrjdic.keys()[keyind]
		print "key:", key
		cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,GTtrjdic[key].frame[0])
		
		im  = plt.imshow(np.zeros_like(frame))
		plt.axis('off')
		
		for tind in range(len(GTtrjdic[key].fullframe)):
			tt = GTtrjdic[key].fullframe[tind]
			print "frame:", tt
			# cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,1.5*(GTtrjdic[key].frame[tind]-GTtrjdic[key].frame[max(0,tind-1)])+frame_here)
			status, frame = cap.read()
			# frame = readVideo(cap,2*(GTtrjdic[key].frame[tind]-GTtrjdic[key].frame[max(0,tind-1)]))
			im.set_data(frame[:,:,::-1])

			"""draw GT rectangle"""
			overlap_upperLX = GTtrjdic[key].fullGTupperL_list[:,0]
			overlap_upperLY = GTtrjdic[key].fullGTupperL_list[:,1]

			overlap_lowerRX = GTtrjdic[key].fullGTLowerR_list[:,0]
			overlap_lowerRY = GTtrjdic[key].fullGTLowerR_list[:,1]

			cnt = [np.int32([[overlap_upperLX[tt-GTtrjdic[key].fullframe[0]],overlap_upperLY[tt-GTtrjdic[key].fullframe[0]]], \
				[overlap_lowerRX[tt-GTtrjdic[key].fullframe[0]],overlap_upperLY[tt-GTtrjdic[key].fullframe[0]]],\
				[overlap_lowerRX[tt-GTtrjdic[key].fullframe[0]],overlap_lowerRY[tt-GTtrjdic[key].fullframe[0]]], \
				[overlap_upperLX[tt-GTtrjdic[key].fullframe[0]],overlap_lowerRY[tt-GTtrjdic[key].fullframe[0]]]])]


			if len(cnt)>0:
				cv2.drawContours(frame, cnt, 0  , (0, 255, 0), 2 )
				im.set_data(frame[:,:,::-1])


			"""draw GT trj"""
			# xx = np.array(GTtrjdic[key].xTrj)[:tind+1]
			# yy = np.array(GTtrjdic[key].yTrj)[:tind+1]
			xx = np.array(GTtrjdic[key].fullxTrj)[:tind+1]
			yy = np.array(GTtrjdic[key].fullyTrj)[:tind+1]

			# xx = np.array(GTtrjdic[key].fullxTrj)[tind]
			# yy = np.array(GTtrjdic[key].fullyTrj)[tind]

			# dots.append(axL.scatter(xx,yy, s=10, color=(color[keyind].T)/255.,edgecolor='none')) 
			# lines = axL.plot(xx,yy,color=(color[keyind].T)/255.,linewidth=1)

			dots.append(axL.scatter(xx,yy, s=10, color='g')) 
			lines = axL.plot(xx,yy,color = 'g',linewidth=1)
			plt.draw()
			plt.show()
			
			"""draw our vehicle obj result"""
			for ind in vehicleCandidates_reorderedInd[key][2]:
				plt.plot(VehicleObjDic[ind].fullxTrj[:tt-VehicleObjDic[ind].frame[0]],VehicleObjDic[ind].fullyTrj[:tt-VehicleObjDic[ind].frame[0]],color = 'r')
			plt.draw()
			plt.show()
			# plt.waitforbuttonpress()
			plt.pause(0.001)
			# if len(dots)>0:
				# for dot in dots:
				# 	dot.remove()
			# axL.lines.pop(0)

		plt.cla()

		plt.draw()
		plt.show()

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
				

				relativeID = cooccur_ran-GTtrjdic[ii].frame[0]
				overlap_upperLX = GTtrjdic[ii].fullGTupperL_list[relativeID,0]
				overlap_upperLY = GTtrjdic[ii].fullGTupperL_list[relativeID,1]

				overlap_lowerRX = GTtrjdic[ii].fullGTLowerR_list[relativeID,0]
				overlap_lowerRY = GTtrjdic[ii].fullGTLowerR_list[relativeID,1]

				"""draw GT and vehicle obj together"""


				# plt.plot(GTtrjdic[ii].fullGTupperL_list[:,0],GTtrjdic[ii].fullGTupperL_list[:,1])
				# plt.plot(GTtrjdic[ii].fullGTLowerR_list[:,0],GTtrjdic[ii].fullGTLowerR_list[:,1])

				# plt.draw()
				# plt.show()
				# # plt.plot(co1X,co1Y,color = 'r')
				# # plt.plot(co2X,co2Y,color = 'g')
				# # plt.draw()
				# # plt.show()
				# # pdb.set_trace()

				# cnt = [np.int32([[overlap_upperLX[0],overlap_upperLY[0]], \
				# 	[overlap_lowerRX[0],overlap_upperLY[0]],\
				# 	[overlap_lowerRX[0],overlap_lowerRY[0]], \
				# 	[overlap_upperLX[0],overlap_lowerRY[0]]])]


				# if len(cnt)>0:
				# 	cap = cv2.VideoCapture(DataPathobj.video)
				# 	st,frame = cap.read()
				# 	cv2.drawContours( frame, cnt, 0  , (0, 255, 0), 2 )
				# 	cv2.imshow('cnt',frame)
				# 	cv2.waitKey(0)

				"""length within the bbox"""
				withinBBox = (co2X<np.maximum(overlap_upperLX,overlap_lowerRX))*\
				(co2X>np.minimum(overlap_upperLX,overlap_lowerRX))* \
				(co2Y<np.maximum(overlap_upperLY,overlap_lowerRY))* \
				(co2Y>np.minimum(overlap_upperLY,overlap_lowerRY))
				vehicleCandidates[ii].append([jj,len(cooccur_ran),distance,np.sum(withinBBox)])



	"""find the same time, assign the one that is within the bbox to GT"""
	vehicleCandidates_reorderedInd ={}

	for ii in GTtrjdic.keys():
		if len(vehicleCandidates[ii])==0:
			continue

		vehicleCandidates_reorderedInd[ii] = []
		overlapInd = np.argsort( np.array(vehicleCandidates[ii])[:,1])[::-1]
		"""relative ind in the list"""
		distInd    = np.argsort( np.array(vehicleCandidates[ii])[:,2])
		withinInd  = np.argsort( np.array(vehicleCandidates[ii])[:,3])[::-1]
		"""only left with vehicles that are within the bbox for more than 1/2 of the overlapping time"""
		# withinInd = withinInd[np.array(vehicleCandidates[ii])[withinInd[:],3]>=0.5*np.array(vehicleCandidates[ii])[withinInd[:],1]]
		withinInd = withinInd[np.array(vehicleCandidates[ii])[withinInd[:],3]>=0.2*np.array(vehicleCandidates[ii])[withinInd[:],1]]
		
		"""corresponding vehicle ID"""
		nearestVehicleID_overlap = np.array(vehicleCandidates[ii])[overlapInd[:]][:,0] 
		nearestVehicleID_dist    = np.array(vehicleCandidates[ii])[distInd[:]][:,0] #5 nearest
		nearestVehicleID_inBB    = np.array(vehicleCandidates[ii])[withinInd[:]][:,0] 

		vehicleCandidates_reorderedInd[ii] = [nearestVehicleID_overlap,nearestVehicleID_dist, nearestVehicleID_inBB,overlapInd,distInd,withinInd]

		# plt.figure()
		# plt.plot(GTtrjdic[ii].fullGTupperL_list[:,0],GTtrjdic[ii].fullGTupperL_list[:,1],color ='g')
		# plt.plot(GTtrjdic[ii].fullGTLowerR_list[:,0],GTtrjdic[ii].fullGTLowerR_list[:,1],color ='g')

		# # plt.plot(GTtrjdic[ii].fullxTrj,GTtrjdic[ii].fullyTrj,color='r')
		# # plt.plot(GTtrjdic[ii].GTupperL_list,GTtrjdic[ii].GTLowerR_list,color ='g')
		# # plt.plot(GTtrjdic[ii].xTrj,GTtrjdic[ii].yTrj,color='r')
		# # for ind in vehicleCandidates_reorderedInd[ii] :
		# # for ind in np.array(vehicleCandidates[ii])[distInd[:10]][:,0]:
		# # for ind in np.array(vehicleCandidates[ii])[overlapInd[:3]][:,0]:
		# for ind in nearestID_inBB[:3]:
		# 	plt.plot(VehicleObjDic[ind].fullxTrj,VehicleObjDic[ind].fullyTrj)
		# 	pdb.set_trace()
		# plt.draw()
		# plt.show()

	# plotGTonVideo(GTtrjdic, vehicleCandidates_reorderedInd)
	missRate, overSegementRate, dist, dist_list= metrics(vehicleCandidates_reorderedInd)
	print "missRate, overSegementRate, dist", missRate, overSegementRate, dist


	plt.figure()
	plt.title('dist_list')
	plt.plot(dist_list)




















