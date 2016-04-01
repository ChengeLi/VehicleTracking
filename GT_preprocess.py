#### preprocess groundth truth
import csv
import pickle as pickle
import numpy as np
import pdb
# from DataPathclass import *
# DataPathobj = DataPath(dataSource,VideoIndex)
# from parameterClass import *
# Parameterobj = parameter(dataSource,VideoIndex)

def GTFromCSV(file, line_limit):
	reader = csv.reader(file)
	# subsampleRate = 4
	GTupperL_list = []
	GTLowerR_list = []
	GTcenterXY_list = []
	frame_idx_list = []

	frame_idx=0
	vehicleInd = 0
	GTtrjdic = {}
	lines = 0
	while lines<line_limit:
		temp = reader.next()
		if np.double(temp[0])<frame_idx: # new car	
			GTtrjdic[vehicleInd] = GTtrj(GTupperL_list,GTLowerR_list,GTcenterXY_list,frame_idx_list)
			frame_idx_list  = []
			GTupperL_list   = []
			GTLowerR_list   = []
			GTcenterXY_list = []
			vehicleInd += 1
		frame_idx = np.double(temp[0])
		GTupperL = np.double(temp[1:3])  #upper left
		GTLowerR = np.double(temp[3:5])  #lower right
		GTcenterXY = np.double(temp[-2:])

		frame_idx_list.append(frame_idx)
		GTupperL_list.append(GTupperL)
		GTLowerR_list.append(GTLowerR)
		GTcenterXY_list.append(GTcenterXY)
		lines+=1
	return GTtrjdic

def readGTdata():

	if dataSource == 'DoT':
		f =  open('/Users/Chenge/Desktop/FirstCanalGroundTruth.csv', 'rb')
		line_limit = 2062 ## number of lines in this csv file
		GTtrjdic = GTFromCSV(f,line_limit)

	elif dataSource == 'Johnson':
		f =  open(DataPathobj.DataPath+'Johnson_00115_ROI_gt.csv', 'rb')
		line_limit = 1128
		GTtrjdic = GTFromCSV(f,line_limit)



	elif dataSource == 'NGSIM':
		f =  open('/Volumes/Transcend/US-101/US-101-MainData/vehicle-trajectory-data/0750am-0805am/trajectories-0750am-0805am.txt', 'rb')
		line_limit = 40000 #only read partial of it
		reader = csv.reader(f)
		GTupperL_list = []
		GTLowerR_list = []
		GTcenterXY_list = []
		frame_idx_list = []

		frame_idx=0
		GTtrjdic = {}
		vehicleInd = np.NaN
		lines = 0
		while lines<line_limit:
			temp = reader.next()
			temp = temp[0].split()
			if np.double(temp[0])!=vehicleInd: # new car
				if not np.isnan(vehicleInd): ##save the last car info
					# plt.plot(np.array(GTcenterXY_list)[:,0],np.array(GTcenterXY_list)[:,1])
					# if laneID==8:
					plt.plot(np.array(GTcenterXY_list)[:,1],np.array(GTcenterXY_list)[:,0])
					plt.draw()
					plt.show()
					GTtrjdic[vehicleInd] = GTtrj(GTupperL_list,GTLowerR_list,GTcenterXY_list,frame_idx_list)

				vehicleInd = np.double(temp[0])	
				frame_idx_list  = []
				GTcenterXY_list = []



			frame_idx = np.double(temp[1])
			"""(X,Y)"""
			# GTcenterXY_local = np.double(temp[4:6])#local XY
			# GTcenterXY_global = np.double(temp[6:8]) #global XY
			GTcenterXY = np.double(temp[4:8]) #both local and global
			""""in feet!"""
			GT_vehicle_len  = np.double(temp[8])
			GT_vehicle_wid  = np.double(temp[9])

			frame_idx_list.append(frame_idx)
			"""bounding box"""
			GTupperL_list.append((np.double(temp[4])-GT_vehicle_len/2, np.double(temp[5])-GT_vehicle_wid/2))
			GTLowerR_list.append((np.double(temp[4])+GT_vehicle_len/2, np.double(temp[5])+GT_vehicle_wid/2))
			GTcenterXY_list.append(GTcenterXY) 


			"""lane ID"""
			laneID = np.double(temp[13])

			lines+=1
	
	return GTtrjdic



class GTtrj(object):
	def __init__(self,GTupperL_list,GTLowerR_list,GTcenterXY_list,frame_idx_list):
		self.GTupperL_list = GTupperL_list
		self.GTLowerR_list = GTLowerR_list
		self.GTcenterXY_list = GTcenterXY_list
		self.time = frame_idx_list



def assign_to_gt_box(trj,GTupperL_list,GTLowerR_list,GTcenterXY_list):
	"""assgin trj to the nearest GroundTruth bounding box"""

	"""find the overlapping time"""


	pass

def clamp(n, minn, maxn):
    # return max(min(maxn, n), minn)
    return (n<=maxn)*(n>=minn)

    
def plotGTonVideo(GTtrjdic):
	if dataSource == 'DoT':
		cap = cv2.VideoCapture(DataPathobj.video)
	elif dataSource == 'Johnson':
		cap = cv2.VideoCapture(DataPathobj.video)
	elif dataSource == 'NGSIM':
		cap = cv2.VideoCapture('/Volumes/Transcend/US-101/US-101-ProcessedVideo-0750am-0805am-Cam1234/sb-camera4-0750am-0805am-processed.avi')

	# cap.set( cv2.cv.CV_CAP_PROP_POS_FRAMES , 0)
	# status, frame = cap.read()
	# plt.imshow(frame[:,:,::-1])
	for ii in GTtrjdic.keys():
		plt.plot(np.array(GTtrjdic[ii].GTcenterXY_list)[1],np.array(GTtrjdic[ii].GTcenterXY_list)[0])
		plt.draw()
		plt.show()



if __name__ == '__main__':
	"""construct GT trj dictionary:"""
	GTtrjdic = readGTdata()
	pdb.set_trace()
	# pickle.dump(GTtrjdic,open(DataPathobj.DataPath+'/GTtrjdictionary_'+	dataSource,'wb'))


	"""gis_cam4_coor"""
	gis_upperL  = (6451788.408,1872650.027)
	gis_uppperR = (6451872.489,1872795.855)
	gis_lowerL  = (6451995.983,1872476.609)
	gis_lowerR  = (6452103.712,1872583.024)

	gis_cam4_coor = np.array([gis_upperL,gis_uppperR,gis_lowerL,gis_lowerR])
	locationValidity = {}
	GTtrjdic_cam4 = {}

	for key in GTtrjdic.keys():
		locationValidity[key] = clamp(np.array(GTtrjdic[key].GTcenterXY_list)[:,2],min(gis_cam4_coor[:,0]),max(gis_cam4_coor[:,0])) *\
		clamp(np.array(GTtrjdic[key].GTcenterXY_list)[:,3],min(gis_cam4_coor[:,1]),max(gis_cam4_coor[:,1]))
		GTcenterXY_list_cam4 = (np.array(GTtrjdic[key].GTcenterXY_list)[:,0][locationValidity[key]],np.array(GTtrjdic[key].GTcenterXY_list)[:,1][locationValidity[key]])
		GTupperL_list_cam4 =np.array(GTtrjdic[key].GTupperL_list)[locationValidity[key]]
		GTLowerR_list_cam4 = np.array(GTtrjdic[key].GTLowerR_list)[locationValidity[key]]
		time_cam4 = np.array(GTtrjdic[key].time)[locationValidity[key]]
		GTtrjdic_cam4[key] = GTtrj(GTupperL_list_cam4,GTLowerR_list_cam4,GTcenterXY_list_cam4,time_cam4)

	plotGTonVideo(GTtrjdic_cam4)


	"""load GT trj dictionary:"""
	# GTtrjdic = pickle.load(open(DataPathobj.DataPath+'/GTtrjdictionary_'+	dataSource,'rb'))


	"""load our system result trj dictionary:"""
	# vctime = pickle.load( open(os.path.join(DataPathobj.dicpath,'final_vctime.p'), "rb" ) )
	# vcxtrj = pickle.load( open(os.path.join(DataPathobj.dicpath,'final_vcxtrj.p'), "rb" ) )
	# vcytrj = pickle.load( open(os.path.join(DataPathobj.dicpath,'final_vcytrj.p'), "rb" ) )	



# plt.plot(np.array(GTtrjdic[4].GTcenterXY_list)[:,0],np.array(GTtrjdic[4].GTcenterXY_list)[:,1])
























