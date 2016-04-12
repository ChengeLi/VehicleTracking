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




if __name__ == '__main__':

	"""load GT trj dictionary:"""
	if dataSource == 'NGSIM':
		filename = DataPathobj.DataPath+'/GTtrjdictionary_'+dataSource+'_cam4.p'
	else:
		filename = DataPathobj.DataPath+'/GTtrjdictionary_'+dataSource+'.p'

	pdb.set_trace()
	GTtrjdic = pickle.load(open(filename,'rb'))
	pdb.set_trace()

	"""load our system result trj dictionary:"""
	VehicleObjDic = pickle.load(open(os.path.join(DataPathobj.pairpath,'VehicleObjDic.p'),'rb'))



	"""find the same time, assign the nearest one to the GT"""







	"""nearest neighbour"""


	pdb.set_trace()
	for ii in GTtrjdic.keys():
		GTtrjdic[ii].time

















