#### preprocess groundth truth
import csv
import pickle as pickle
import numpy as np
import pdb
from scipy.interpolate import interp1d

"""thre's bug in cap.set, try loopy reading instead"""
def readVideo(cap,subSampRate):
    """when read video in a loop, every subSampRate frames"""
    status, frame = cap.read()  
    for ii in range(subSampRate-1):
        status, frameskip = cap.read()
    return frame



def GTFromCSV(file, line_limit,Gdd_img_list_interval):
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
		if np.double(temp[0])<frame_idx or lines==line_limit-1: # new car	
			print "length", len(frame_idx_list)
			GTtrjdic[vehicleInd] = GTtrj(GTupperL_list,GTLowerR_list,GTcenterXY_list,frame_idx_list,vehicleInd)
			frame_idx_list  = []
			GTupperL_list   = []
			GTLowerR_list   = []
			GTcenterXY_list = []
			vehicleInd += 1
		frame_idx = np.double(temp[0])
		frame_idx_real = 0+Gdd_img_list_interval*frame_idx
		# GTupperL = np.double(temp[1:3])  #upper left
		# GTLowerR = np.double(temp[3:5])  #lower right
		# GTcenterXY = np.double(temp[-2:])
		

		GTcenterXY = np.double(temp[-2:])

		frame_idx_list.append(frame_idx_real)
		GTupperL_list.append((np.double(temp[1]), np.double(temp[2])))
		GTLowerR_list.append((np.double(temp[3]), np.double(temp[4])))


		GTcenterXY_list.append(GTcenterXY)
		lines+=1
	return GTtrjdic

def readGTdata():

	if dataSource == 'DoT':
		f =  open('/Users/Chenge/Desktop/FirstCanalGroundTruth.csv', 'rb')
		line_limit = 2062 ## number of lines in this csv file
		# GTtrjdic = GTFromCSV(f,line_limit, ???)
		GTtrjdic = GTFromCSV(f,line_limit, 1) #what's the interval for the canal video

	elif dataSource == 'Johnson':
		f =  open(os.path.join(DataPathobj.DataPath,'Johnson_00115_ROI_gt.csv'), 'rb')
		line_limit = 1128
		GTtrjdic = GTFromCSV(f, line_limit,1)

	elif dataSource == 'NGSIM':
		
		"""read GT from the csv"""
		f =  open(os.path.join(DataPathobj.DataPath,'NGSIM_gt.csv'), 'rb')
		line_limit = 18

		# f =  open(os.path.join(DataPathobj.DataPath,'NGSIM_gt_jiaxu.csv'), 'rb')
		# line_limit = 527

		GTtrjdic = GTFromCSV(f,line_limit,1)


		"""read GT from the txt"""
		# f =  open('/Volumes/Transcend/US-101/US-101-MainData/vehicle-trajectory-data/0750am-0805am/trajectories-0750am-0805am.txt', 'rb')
		# line_limit = 40000 #only read partial of it
		# reader = csv.reader(f)
		# GTupperL_list = []
		# GTLowerR_list = []
		# GTcenterXY_list = []
		# frame_idx_list = []

		# frame_idx=0
		# GTtrjdic = {}
		# vehicleInd = np.NaN
		# lines = 0
		# while lines<line_limit:
		# 	temp = reader.next()
		# 	temp = temp[0].split()
		# 	if np.double(temp[0])!=vehicleInd: # new car
		# 		if not np.isnan(vehicleInd): ##save the last car info
		# 			# plt.plot(np.array(GTcenterXY_list)[:,0],np.array(GTcenterXY_list)[:,1])
		# 			# if laneID==8:
		# 			plt.plot(np.array(GTcenterXY_list)[:,1],np.array(GTcenterXY_list)[:,0])
		# 			plt.draw()
		# 			plt.show()
		# 			GTtrjdic[vehicleInd] = GTtrj(GTupperL_list,GTLowerR_list,GTcenterXY_list,frame_idx_list,vehicleInd)
		# 		vehicleInd = np.double(temp[0])	
		# 		frame_idx_list  = []
		# 		GTcenterXY_list = []



		# 	frame_idx = np.double(temp[1])
		# 	"""(X,Y)"""
		# 	# GTcenterXY_local = np.double(temp[4:6])#local XY
		# 	# GTcenterXY_global = np.double(temp[6:8]) #global XY
		# 	GTcenterXY = np.double(temp[4:8]) #both local and global
		# 	""""in feet!"""
		# 	GT_vehicle_len  = np.double(temp[8])
		# 	GT_vehicle_wid  = np.double(temp[9])

		# 	frame_idx_list.append(frame_idx)
		# 	"""bounding box"""
		# 	GTupperL_list.append((np.double(temp[4])-GT_vehicle_len/2, np.double(temp[5])-GT_vehicle_wid/2))
		# 	GTLowerR_list.append((np.double(temp[4])+GT_vehicle_len/2, np.double(temp[5])+GT_vehicle_wid/2))
		# 	GTcenterXY_list.append(GTcenterXY) 


		# 	"""lane ID"""
		# 	laneID = np.double(temp[13])
		# 	lines+=1
	
	return GTtrjdic




"""use the same attribute name as class VehicleObj, easier to compare"""
class GTtrj(object):
	def __init__(self,GTupperL_list,GTLowerR_list,GTcenterXY_list,frame_idx_list,ID):
		"""attributes are the same as the vehicle obj class, except that GT has bounding box info """
		self.VehicleID = ID
		self.xTrj      = np.array(GTcenterXY_list)[:,0]
		self.yTrj      = np.array(GTcenterXY_list)[:,1]
		self.frame     = np.int32(frame_idx_list)
		self.vel       = [] 
		self.pos       = [] 
		self.status    = 1   # 1: alive  2: dead
		self.Xdir      = []
		self.Ydir      = []

		self.GTupperL_list = np.array(GTupperL_list)
		self.GTLowerR_list = np.array(GTLowerR_list)
		# self.GTcenterXY_list = GTcenterXY_list

		fullFrameLen = self.frame[-1]-self.frame[0]+1
		if fullFrameLen>len(self.frame):
			"""interpolate gt vector to full time resolution"""
			# interpolationMethod = 'cubic'
			interpolationMethod = 'linear'

			fx = interp1d(self.frame,self.xTrj, kind=interpolationMethod)
			fy = interp1d(self.frame,self.yTrj, kind=interpolationMethod)
			self.fullxTrj  = fx(range(self.frame[0],self.frame[-1]+1,1))
			self.fullyTrj  = fy(range(self.frame[0],self.frame[-1]+1,1))

			fulx = interp1d(self.frame,np.array(self.GTupperL_list)[:,0], kind=interpolationMethod)
			fuly = interp1d(self.frame,np.array(self.GTupperL_list)[:,1], kind=interpolationMethod)
			flrx = interp1d(self.frame,np.array(self.GTLowerR_list)[:,0], kind=interpolationMethod)
			flry = interp1d(self.frame,np.array(self.GTLowerR_list)[:,1], kind=interpolationMethod)

			self.fullframe = range(self.frame[0],self.frame[-1]+1,1)
			self.fullGTupperL_list  = np.array([fulx(self.fullframe),fuly(self.fullframe)]).T
			self.fullGTLowerR_list  = np.array([flrx(self.fullframe),flry(self.fullframe)]).T

			# plt.plot(self.fullGTupperL_list,self.fullGTLowerR_list)
			# plt.draw()
			# pdb.set_trace()
			# plt.show()

		else:
			self.fullxTrj = self.xTrj
			self.fullyTrj = self.yTrj
			self.fullGTupperL_list = self.GTupperL_list 
			self.fullGTLowerR_list = self.GTLowerR_list 
			self.fullframe = self.frame




def clamp(n, minn, maxn):
    # return max(min(maxn, n), minn)
    return (n<=maxn)*(n>=minn)

    
def plotGTonVideo(GTtrjdic):
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
	im  = plt.imshow(np.zeros_like(frame))
	color = np.array([np.random.randint(0,255) for _ in range(3*len(GTtrjdic))]).reshape(len(GTtrjdic),3)
	plt.axis('off')
	dots = []
	for keyind in range(len(GTtrjdic.keys())):
		key = GTtrjdic.keys()[keyind]
		print "key:", key
		cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,GTtrjdic[key].frame[0])
		# pdb.set_trace()
		# for tind in range(len(GTtrjdic[key].frame)+1):
		# 	tt = GTtrjdic[key].frame[tind]
		for tind in range(len(GTtrjdic[key].fullframe)):
			tt = GTtrjdic[key].fullframe[tind]
			print "frame:", tt
			# cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,1.5*(GTtrjdic[key].frame[tind]-GTtrjdic[key].frame[max(0,tind-1)])+frame_here)
			status, frame = cap.read()
			# frame = readVideo(cap,2*(GTtrjdic[key].frame[tind]-GTtrjdic[key].frame[max(0,tind-1)]))
			im.set_data(frame[:,:,::-1])

			# xx = np.array(GTtrjdic[key].xTrj)[:tind+1]
			# yy = np.array(GTtrjdic[key].yTrj)[:tind+1]
			xx = np.array(GTtrjdic[key].fullxTrj)[:tind+1]
			yy = np.array(GTtrjdic[key].fullyTrj)[:tind+1]

			# xx = np.array(GTtrjdic[key].fullxTrj)[tind]
			# yy = np.array(GTtrjdic[key].fullyTrj)[tind]

			dots.append(axL.scatter(xx,yy, s=10, color=(color[keyind].T)/255.,edgecolor='none')) 
			lines = axL.plot(xx,yy,color=(color[keyind].T)/255.,linewidth=1)
			plt.draw()
			plt.show()
			plt.pause(0.001)
			# plt.waitforbuttonpress()

	for i in dots:
		i.remove()

	try:
		axL.lines.pop(0)
	except:
		pass



if __name__ == '__main__':
	from DataPathclass import *
	DataPathobj = DataPath(dataSource,VideoIndex)
	from parameterClass import *
	Parameterobj = parameter(dataSource,VideoIndex)


	"""construct GT trj dictionary:"""
	GTtrjdic = readGTdata()
	pickle.dump(GTtrjdic,open(DataPathobj.pairpath+'/GTtrjdictionary_'+	dataSource+'.p','wb'))
	plotGTonVideo(GTtrjdic)

	# for ii in GTtrjdic.keys():
	# 	plot(GTtrjdic[ii].fullxTrj, GTtrjdic[ii].fullyTrj)


	"""if using the gt ngsim provided, we need to select out the cam-4 region"""
	# if dataSource == 'NGSIM':
	# 	"""gis_cam4_coordinates"""
	# 	gis_upperL  = (6451788.408,1872650.027)
	# 	gis_uppperR = (6451872.489,1872795.855)
	# 	gis_lowerL  = (6451995.983,1872476.609)
	# 	gis_lowerR  = (6452103.712,1872583.024)

	# 	gis_cam4_coor = np.array([gis_upperL,gis_uppperR,gis_lowerL,gis_lowerR])
	# 	locationValidity = {}
	# 	GTtrjdic_cam4 = {}

	# 	for key in GTtrjdic.keys():
	# 		locationValidity[key] = clamp(np.array(GTtrjdic[key].GTcenterXY_list)[:,2],min(gis_cam4_coor[:,0]),max(gis_cam4_coor[:,0])) *\
	# 		clamp(np.array(GTtrjdic[key].GTcenterXY_list)[:,3],min(gis_cam4_coor[:,1]),max(gis_cam4_coor[:,1]))
	# 		GTcenterXY_list_cam4 = (np.array(GTtrjdic[key].GTcenterXY_list)[:,0][locationValidity[key]],np.array(GTtrjdic[key].GTcenterXY_list)[:,1][locationValidity[key]])
	# 		GTupperL_list_cam4 =np.array(GTtrjdic[key].GTupperL_list)[locationValidity[key]]
	# 		GTLowerR_list_cam4 = np.array(GTtrjdic[key].GTLowerR_list)[locationValidity[key]]
	# 		time_cam4 = np.array(GTtrjdic[key].time)[locationValidity[key]]
	# 		GTtrjdic_cam4[key] = GTtrj(GTupperL_list_cam4,GTLowerR_list_cam4,GTcenterXY_list_cam4,time_cam4)

	# 	plotGTonVideo(GTtrjdic_cam4)
	# 	pickle.dump(GTtrjdic_cam4,open(DataPathobj.DataPath+'/GTtrjdictionary_'+dataSource+'_cam4.p','wb'))


















