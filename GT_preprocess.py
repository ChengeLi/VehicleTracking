#### preprocess groundth truth
import csv
import pickle as pickle
import numpy as np
import pdb

def readGTdata():
	f =  open('/Users/Chenge/Desktop/GroundTruth.csv', 'rb')
	reader = csv.reader(f)
	subsampleRate = 4

	# frame_idx = 10280
	GTupperL_list = []
	GTLowerR_list = []
	GTcenterXY_list = []
	frame_idx_list = []
	vehicleInd_list = []

	frame_idx=0
	vehicleInd = 0
	GTtrjdic = {}
	while frame_idx<4000:
		temp = reader.next()
		# for temp in reader:
		# pdb.set_trace()
		
		if np.double(temp[0])<frame_idx: # new car	
			GTtrjdic[vehicleInd] = GTtrj(GTupperL_list,GTLowerR_list,GTcenterXY_list,frame_idx_list)
			frame_idx_list  = []
			GTupperL_list   = []
			GTLowerR_list   = []
			GTcenterXY_list = []
			vehicleInd_list = []
			vehicleInd += 1

		frame_idx = np.double(temp[0])
		GTupperL = np.double(temp[1:3])  #upper left
		GTLowerR = np.double(temp[3:5])  #lower right
		GTcenterXY = np.double(temp[-2:])
		frame_idx_list.append(frame_idx)
		GTupperL_list.append(GTupperL)
		GTLowerR_list.append(GTLowerR)
		GTcenterXY_list.append(GTcenterXY)
		return GTtrjdic



class GTtrj(object):
	def __init__(self,GTupperL_list,GTLowerR_list,GTcenterXY_list,frame_idx_list):
		self.GTupperL_list = GTupperL_list
		self.GTLowerR_list = GTLowerR_list
		self.GTcenterXY_list = GTcenterXY_list
		self.time = frame_idx_list





def assign_to_gt_box(trj,GTupperL_list,GTLowerR_list,GTcenterXY_list):
	'assgin trj to the nearest GroundTruth bounding box'

	'find the overlapping time'
	# trj.frame_idx & 


	pass

	'interpolate trj'
	xtrj = GTupperL[:,0]
	ytrj = GTupperL[:,1]
	ytrjFull = np.interp(np.range(len(max(xtrj))), xtrj, ytrj)




if __name__ == '__main__':
	GTtrjdic = readGTdata()
	pdb.set_trace()
	GTtrjobj = GTtrj(GTupperL_list,GTLowerR_list,GTcenterXY_list,frame_idx_list,vehicleInd_list)
	pickle.dump(GTtrjobj,open('./Canal_GTtrjobj','wb'))































